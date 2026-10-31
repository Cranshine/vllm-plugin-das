# SPDX-License-Identifier: Apache-2.0
"""VMFault forensic probes for the NC/dense FlashMLA fast paths.

R33 branch-probe track (perf/kimik3-d-moe-fork-vmfaultfix). Plain-stdout
one-line JSON evidence around every expanded/clamped block-table fill and
every fp8 MLA launch of the suspect region, so a bs64 opencompass-full
reproduction leaves a per-step launch geometry record that survives the
crash itself.

Hard rules (cudagraph-capture-safety):
  - NEVER sync, allocate, or copy inside CUDA capture. Every helper here
    checks ``torch.cuda.is_current_stream_capturing()`` first and becomes a
    host-only counter bump (or nothing at all) while capturing.
  - NEVER mutate model state, never raise: a broken probe must not change
    service behavior. All probe failures are swallowed.
  - The device->host audit copies are plain blocking ``.to("cpu")`` calls
    enqueued on the running stream OUTSIDE capture; each costs one
    pipeline flush per launch, which is accepted on this debug branch.

Env knobs:
  VLLM_HCU_VMFPROBE       "1" (default) on this debug branch; "0" disables
                          everything including the atexit summary.
  VLLM_HCU_VMFPROBE_SYNC  "1" (default) perform device->host table audits;
                          "0" logs host-side launch metadata only.
  VLLM_HCU_VMFPROBE_EVERY audit every Nth fill site hit (default 1 = all).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Any

__all__ = [
    "capturing",
    "mark",
    "audit_table",
    "audit_seq_lens",
    "finish_probe_session",
]


def _env_on(name: str, default: str = "1") -> bool:
    v = os.environ.get(name, default)
    return str(v).lower() not in ("0", "", "false", "no", "off")


PROBE_ON = _env_on("VLLM_HCU_VMFPROBE", "1")
SYNC_ON = _env_on("VLLM_HCU_VMFPROBE_SYNC", "1")
try:
    EVERY = max(1, int(os.environ.get("VLLM_HCU_VMFPROBE_EVERY", "1")))
except ValueError:
    EVERY = 1

_SELF_SHA = ""
try:
    with open(__file__, "rb") as _f:
        _SELF_SHA = hashlib.sha256(_f.read()).hexdigest()[:16]
except OSError:
    pass

_SHA_CACHE: dict[str, str] = {}


def file_sha(path: str) -> str:
    """Short sha256 of a source file (line-fingerprint header discipline)."""
    try:
        if path not in _SHA_CACHE:
            with open(path, "rb") as f:
                _SHA_CACHE[path] = hashlib.sha256(f.read()).hexdigest()[:16]
        return _SHA_CACHE[path]
    except OSError:
        return "unknown"

_T0 = time.monotonic()
STATS: dict[str, Any] = {
    "site_hits": 0,
    "audits": 0,
    "raw_oob": 0,       # audits where raw table had ids >= bound
    "post_oob": 0,      # audits where post-clamp table still had ids >= bound
    "max_raw_id": -1,
    "max_seq": -1,
    "first_raw_oob": None,
}


def capturing() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        # No CUDA context yet (import time) -> treat as not capturing; any
        # later real audit re-checks.
        return False


def _emit(kind: str, where: str, **kw: Any) -> None:
    try:
        rec = {
            "t": round(time.monotonic() - _T0, 3),
            "pid": os.getpid(),
            "k": kind,
            "w": where,
            "sha": _SELF_SHA,
        }
        rec.update(kw)
        sys.stdout.write("VMFPROBE " + json.dumps(rec, default=str) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def mark(where: str, **kw: Any) -> None:
    """Host-only metadata line. Safe anywhere (including capture)."""
    if not PROBE_ON:
        return
    _emit("mark", where, cap=capturing(), **kw)


def audit_table(
    where: str,
    table2d_used,
    bound: int,
    num_rows: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Sync-audit the leading ``num_rows`` rows of a page table.

    ``bound`` is the legal exclusive page-id bound (pool pages or cache
    page count). Logs raw min/max/OOB count both BEFORE any clamp is
    enqueued (call pre-clamp) and AFTER (call post-clamp); ``phase``
    distinguishes the two.
    """
    if not PROBE_ON:
        return
    STATS["site_hits"] += 1
    hit = STATS["site_hits"]
    if capturing() or not SYNC_ON or (hit % EVERY) != 0:
        # Capture context: host-only breadcrumb, no GPU interaction at all.
        _emit("site", where, cap=True, hit=hit, rows=int(num_rows))
        return
    try:
        t = table2d_used[:num_rows].detach().to("cpu", non_blocking=False)
        flat = t.reshape(-1).tolist()
        rows = int(num_rows)
        big = bound
        oob_idx = [i for i, v in enumerate(flat) if v >= big]
        lo = min(flat) if flat else -1
        hi = max(flat) if flat else -1
        phase = str((extra or {}).get("phase", "?"))
        is_oob = bool(oob_idx)
        if phase == "raw":
            STATS["max_raw_id"] = max(STATS["max_raw_id"], hi)
            if is_oob:
                STATS["raw_oob"] += 1
                if STATS["first_raw_oob"] is None:
                    STATS["first_raw_oob"] = {"t": hit, "w": where}
        else:
            STATS["max_post_id"] = max(STATS.get("max_post_id", -1), hi)
            if is_oob:
                STATS["post_oob"] += 1
        kw = dict(extra or {})
        _emit(
            "audit",
            where,
            cap=False,
            hit=hit,
            rows=rows,
            bound=big,
            raw_min=lo,
            raw_max=hi,
            n_oob=len(oob_idx),
            oob_cells=oob_idx[:8],
            **kw,
        )
    except Exception as exc:  # probe must never alter behavior
        _emit("audit_err", where, err=f"{type(exc).__name__}: {exc}")


def audit_seq_lens(
    where: str,
    seq_lens_used,
    extra: dict[str, Any] | None = None,
) -> None:
    """Sync-audit the seq_lens the kernel will walk against."""
    if not PROBE_ON:
        return
    if capturing() or not SYNC_ON:
        return
    try:
        t = seq_lens_used.detach().to("cpu", non_blocking=False)
        flat = t.reshape(-1).tolist()
        smax = max(flat) if flat else -1
        STATS["max_seq"] = max(STATS["max_seq"], smax)
        kw = dict(extra or {})
        _emit("seqlens", where, smax=smax, smin=min(flat) if flat else -1, **kw)
    except Exception as exc:
        _emit("seqlens_err", where, err=f"{type(exc).__name__}: {exc}")


def finish_probe_session() -> None:
    if not PROBE_ON:
        return
    _emit("session_end", "atexit", **STATS)


try:
    import atexit as _atexit

    _atexit.register(finish_probe_session)
except Exception:
    pass
