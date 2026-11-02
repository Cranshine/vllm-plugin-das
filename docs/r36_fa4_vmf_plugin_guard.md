# R36 — 插件侧 FA4 funnel `max_seqlen_q` 收缩守卫（VMFault 修复）

R34（dist 漏斗就地 pad 64→128, commit 26183ee）的插件化继任：不改 dist 文件、不改
flash-attention-das 源码，只改 vllm_hcu 插件；目标是不用 pad 也安全，并封 R34 的三个看口。

## 源码定罪（不变，本修复的目标就是它的推论）

- `flash-attention-das csrc/flash_attn/fwd_launch_template.h:153`
  `is_even_MN = !is_varlen && seqlen_k % kBlockN == 0 && seqlen_q % kBlockM == 0`
  —— launcher 收到 `seqlen_q` 为 kBlockM=128 倍数时全局判据为真。
- `csrc/flash_attn/fwd_epilogue_gfx938.h:39-45`（O 行）/ `csrc/flash_attn/fwd_epilogue.h:135-143`（LSE 行）
  —— `if constexpr` 裁掉行守卫后 kernel 无条件写满 128 行。实际 O=64 行 ×12 头 ×192×2B=288KiB，
  写 128 行越界 +66KiB → VMFault。
- dist 漏斗 `vllm/vllm_flash_attn/flash_attn_interface.py::flash_attn_varlen_func`
  （上游已核对：38 个参数全部 `POSITIONAL_OR_KEYWORD`；fa_version=4 分支把 `max_seqlen_q`
  逐字转发给 `vllm.vllm_flash_attn.cute.interface._flash_attn_fwd`）。
  凶手实例 fa_version=4、out=None、Is_even_MN=true、Is_Varlen=false、Layout=1(BSHD,895μs)
  与 Layout=0(52μs) 两实例同病。

## 方案裁决：X 首选（零 pad），以插件 monkey-patch 落在漏斗上

只要 launcher 的 `seqlen_q % 128 != 0`，`is_even_MN=false`，行守卫编译进 kernel，
天然安全且零 pad 开销。插件把漏斗的 `max_seqlen_q` 收缩到「鸽笼上界」：

```
row_bound = total_q - batch + 1        # 每请求查询行数的最坏上界（其余请求各占 >=1 行）
row_bound < budget  →  shrunk = row_bound            （row_bound % 128 != 0 时）
                        shrunk = row_bound + 127     （row_bound 是 128 倍数时，仍 < budget、
                                                      仍 >= row_bound、仍非 128 倍数）
其余（row_bound == budget 等）→ 原样放行（此时 kernel 写不满真实行数，无从越界）
```

不变量（`tests` 单测已跑绿）：
`shrunk % 128 != 0`，`true_max <= shrunk < budget`，`ceil(shrunk/128) >= ceil(true_max/128)`
—— grid m 维只增不减，多出的 m-block 对同一 metadata 计算、被守卫遮蔽的行不产生写入。

### 为什么插件侧改这里等价安全

1. 漏斗是 dist 所有调用形态去 vendor launcher 的唯一皱缩点（fa_version/out/descale 都在此）；
   它把 `max_seqlen_q` 逐字下发，所以在漏斗上收缩 == 源码推论要求的「launcher 收到真实 seqlen_q」。
2. patch 经 `ExactImportCoordinator` 在**漏斗模块 import 完成时**安装；之后任何
   `from vllm.vllm_flash_attn import flash_attn_varlen_func` 的 dist 后端都绑定到包装器
   （模块完成序保证），等价于 R34 的就地改动，但零 dist 文件接触。
3. capture 安全：包装器只做 python int/shape 运算（无 tensor 写、无分配、无 D2H sync），
   capture 内合法；capture 时炸入的是 host 常量分支，replay 期行为一致。
4. 探针纪律：签名漂移 → 记 warning + 软降为 no-op（不 fail startup）；包装器内部任何异常
   → 原参数直调原始函数，绝不新增 crash 路径。

## 文件清单

| 文件 | 变更 |
| --- | --- |
| `vllm_hcu/patch/worker/core_fix/patch_flash_attn_vmf_guard.py` | 新增，308 行，PATCH_ID=`worker.core_fix.flash_attn_vmf_guard` |
| `vllm_hcu/patch/worker/__init__.py` | `_CORE_CALLBACKS` 追加 1 行登记（feature=always） |
| `docs/r36_fa4_vmf_plugin_guard.diff` | 本变更 unified diff（329 行） |
| `docs/r36_fa4_vmf_plugin_guard.md` | 本 README |

## 安装

插件树 editable 安装，代码即部署：

1. 本树 `perf/kimik3-d-moe-fork-vmfaultfix` 增补上述文件（当前未提交，diff 见 docs/*.diff）。
2. 按 sugon-vllm-hcu-migrate SOP 同步到节点容器（`git archive` 打包 → NFS → 原 tree 原子替换
   → `/opt/vllm_hcu_src` 符号链接缓存校验 import）。
3. 起服务冒烟，日志应出现：
   `[HCU FA4 VMF GUARD] armed on vllm.vllm_flash_attn.flash_attn_interface.flash_attn_varlen_func (R36)`
   以及第一次收缩时的
   `[HCU FA4 VMF GUARD] shrunk launcher max_seqlen_q 128 -> 64 ...`。

运行期计数器（无设备同步，任意 worker 内读取）：

```bash
python -c "from vllm_hcu.patch.worker.core_fix import patch_flash_attn_vmf_guard as g; print(dict(g.guard_counters()))"
# {'total': N, 'shrunk': k, 'passthrough': N-k, 'errors': 0}
```

## 回滚

- 单文件回滚：`git revert <commit>`（或删除 patch 文件 + `_CORE_CALLBACKS` 里那一行登记即可，
  装载器按模块名查找，无其它状态）。
- 应急回退不换树：在节点容器里 `python - <<'EOF'`把漏斗模块的 `_vllm_hcu_fa4_vmf_guard_applied`
  置真并还原符号恢复原始函数——不推荐，正常走 git。

## MRE 验证方案（等卡空）

目的：证明方案 X 的守卫分支被激活（kernel 符号 `Is_even_MN=false` + 故障消失 + dur 态样）。

| # | 形状组合 | 预期 |
| --- | --- | --- |
| 1（凶手复刻） | b=1，cu_seqlens_q=[0,64]，q 64 行，funnel 收到 max_seqlen_q=128（R34 之前的膨胀值），fa_version=4, out=None | 守卫激活：launcher 实际 seqlen_q=64 → `Is_even_MN=false`，无 VMFault；fa4 dur 无 895μs 越界态样 |
| 2（均分批） | b=2，行 [64,64]，total=128，budget=128 | 收缩 127（`total`=偶数但 per-req 不等长），不 fault |
| 3（真 128 对照） | b=1，total=128，budget=128 | passthrough（计数器 shrunk 不增），`Is_even_MN=true` 但行数恰等，无害且性能态样（52μs Layout=0 实例）不变 |
| 4（R34 看口①） | 同 #1 但 out≠None（调用方自备 buffer） | 收缩同样生效（wrapper 不看出口） |
| 5（R34 看口②） | 同 #1 但 fa_version=2/3 | 收缩同样生效（wrapper 版本无关） |
| 6（LSE 同病） | #1 + return_softmax_lse=True | `fwd_epilogue.h:135-143` LSE 行守卫同因激活 |
| 7（负对照） | 回滚本补丁跑 #1 | 复现 VMFault（HSA queue 队头 `flash_fwd_kernel_gfx938`，fault VA = draft k/O slab +≤2MiB） |

判据：#1/#2/#4/#5/#6 无 fault 且 HSA 日志中该 kernel 的 `Is_even_MN=false`；#7 复现；#3 性能回归差额 <1%。

## R34 去留结论

R34（dist 文件就地 pad）**保留但前置短路**——dist 树不在本地改不了，节点侧在原 R34 pad 块
头部加：

```diff
--- a/vllm/vllm_flash_attn/flash_attn_interface.py   # 节点 dist 树，非本仓库
+++ b/vllm/vllm_flash_attn/flash_attn_interface.py
@@ <R34 pad 块>
+        # R36 short-circuit: the plugin-side funnel wrapper shrinks the
+        # launcher budget below the packed-row bound, so the epilogue row
+        # guard re-arms without padding.  Skip the 64->128 pad whenever the
+        # guard is already provably armed or the budget is unreachable.
+        _row_bound = q.shape[0] - (cu_seqlens_q.shape[0] - 1) + 1
+        if max_seqlen_q % 128 != 0 or _row_bound < max_seqlen_q:
+            return <原始未 pad 执行路径>
```

即：只要「guard 已武装」或「预算不可达（R36 必收缩）」，pad 整体跳过 → R34 的 pad 开销归零、
其 out≠None / 非 FA4 / dist 直改三个看口由 R36 补位。若节点回归确认 #1–#6 全绿，
可直接整段 revert R34（26183ee），R36 单支柱兜底。

## Commit message

```
fix(attention): shrink FA funnel max_seqlen_q to the packed row bound so the epilogue row guard compiles in

Why: the gfx938 flash-attention-das launcher disarms its epilogue row
guard whenever seqlen_q is a kBlockM multiple (fwd_launch_template.h:153,
is_even_MN = !is_varlen && seqlen_k % kBlockN == 0 && seqlen_q % kBlockM
== 0); the kernel then stores full kBlockM=128 rows unconditionally
(fwd_epilogue_gfx938.h:39-45, fwd_epilogue.h:135-143), so a dist-side
funnel budget of 128 over a 64-row packed O walked +66 KiB past the slab
and VMFaulted (12 q-heads x d192 x bf16).

What: wrap vllm.vllm_flash_attn.flash_attn_interface.flash_attn_varlen_
func from the plugin via the exact-import coordinator (no dist file and
no flash-attention-das source touched).  When total_q - batch + 1 < the
launcher budget, replace the budget with that pigeonhole bound (bumped
by 127 when the bound itself is a 128 multiple); any non-multiple budget
re-arms the guard, grid m only grows (ceil(shrunk/128) >=
ceil(true_max/128)) and masked rows store nothing.  Pure host-side shape
arithmetic: capture-safe, no allocation, no D2H sync.  Signature audit
degrades to a recorded no-op on drift; the wrapper never adds a crash
path.  Covers both causal-true/flattened draft calls and every
fa_version/out combination, closing R34's three gaps.

Validation: unit sweep (killer b=1/64 rows -> budget 64; even-total
[64,64] -> 127; true-128 -> passthrough; multiple-bound bump invariants)
green on Windows; node MRE pending GPU availability per
docs/r36_fa4_vmf_plugin_guard.md (#1-#7 matrix: expect Is_even_MN=false,
no VMFault, dur signature back to the 52us instance).
```
