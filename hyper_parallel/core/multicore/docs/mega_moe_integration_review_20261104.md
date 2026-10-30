# MegaMoe 整合审查与边界正确性修复（2026-11-04）

本轮在 `291bd7437bd4a91e95a0e9dba71aff5c1669be65` 上审查路由端点、共享专家、
缓冲区所有权和设备执行边界，继续处理上一轮报告列出的冷启动小形状、零接收 rank 和共享权重梯度精度问题。
上一轮[性能报告](mega_moe_full_fusion_results_20261103.md)保留其当时的数据和限制，本报告记录后续定位与验证。

本轮 BF16 已验证范围内，冷启动小形状与空接收 rank 的越界已修复；保留图重复反向逐位一致，
共享权重梯度通过原有外部阈值，正式形状 dW1 的超阈值元素由每 rank 31–43 个降为零。
主机回归 296 项、1696 子用例通过；随机与空接收路由的两套 64 组合均通过。


## 审查结论与修改

优先级最高的三项问题分别是原生归约的设备上下文、共享分支的舍入语义、间接指针访问的 allocator 生命周期。
前两项已用原版本独立复现并定位到具体边界；第三项是审查发现的所有权缺口。
本轮不改图调度、不改公共 ABI，不增加 HBM 工作区；保留上一轮已验证的接收区复用和图内分支合并。

### Router 权重梯度归约缺少独立的 TPipe 生命周期

`RunTopKWeightGrad` 使用 CANN `ReduceSum`，随后通过标量读取 UB 中的归约结果。
原实现没有创建任务自身的 `TPipe`，却使用了读取 `GetTPipePtr()` 的同步函数。
CANN 9.1 的归约实现内部也通过该指针获取事件，因此仅替换外层同步调用仍会失败。
运行过其他原语的 worker 可能偶然拥有可用状态，这解释了“先执行完整开关矩阵可以通过，冷启动只执行对照和全融合会失败”的执行顺序敏感现象。

修复在整个归约任务内创建和销毁局部 `TPipe`。既有计算、归约顺序、task/event 数量和终止条件保持不变。
零接收 rank 的路由计算任务可以没有实际 token，然而该 rank 仍须计算自己发出 token 的 Router 权重梯度；
不能通过跳过归约或补 warmup 规避这个边界。

定位证据：

- 原版本冷启动随机小形状和零接收 rank 均复现 `507015`、AIV scalar internal buffer out of bounds。
- 在每次 forward/backward 后显式同步，证明错误可以出现在第一次融合 backward；此前在重复 backward 检查处出现的 traceback 是异步暴露位置。
- 仅将 Router 权重梯度任务的有效行数设为零、保留其他任务和事件，可完成冷启动及重复反向。
  此为排错实验，数值比较被禁用，未作为正确性通过证据，也未保留在交付代码中。
- 仅替换归约之后的外层同步未能解决错误，随后检查安装的 CANN 源码确认归约内部也使用 `GetTPipePtr()`。

### 共享 SwiGLU 省略了外部路径的 BF16 舍入

外部共享专家的前向是 `linear → silu → mul → linear`；SiLU 的输出先舍入为输入 dtype，再参与乘法。
原生 SwiGLU 则在 FP32 中合并 SiLU 和乘法，最后只舍入一次。
反向同样存在差别：外部乘法产生的 `dSiLU` 先舍入，再进入 SiLU backward；`dUp` 消费的 SiLU 值也已舍入。
这些差异随后进入权重梯度的长归约，在接近零的元素上违反原有点对点阈值。

修复为共享分支显式保留这三个舍入边界，仍在原生 kernel 的既有 UB 内完成计算；
没有增加全尺寸 HBM 中间张量。共享任务通过既有间接 tensor 类型选择该语义，
路由专家继续采用其既有融合 SwiGLU 数学定义。未修改公共原生参数布局。

修复前逐边界测量：每 rank 的 shared up projection 与 dActivation 均逐位一致；
activation 最大差为 `0.0625`，dUp 最大差为 `0.001953125`，shared W1 梯度 RMS 约 `0.00329`，
每 rank 有 `31–43 / 18,350,080` 个元素违反 `rtol=0.05, atol=0.01`。
这些数值来自本轮独立复现；与上一轮的具体违规元素数略有差别，首次分歧边界和误差量级一致。

回归移除使用旧原生权重梯度替代外部判据的豁免，所有输出和梯度都必须通过原有外部参考阈值。
重复反向和 pending forward 的反序 backward 继续要求逐位一致。

### 间接共享张量的 stream 生命周期

原生 schema 只能看见路由操作数，共享张量通过指针表访问。
原先仅对指针表和共享 tiling 登记 stream，缺少对间接 payload 的登记。
现将全部间接输入、临时区和输出登记到实际消费 stream，避免它们从其他 stream 创建后被缓存分配器提前复用。
该项是所有权边界补强，不将其混同于前述已经复现的 AIV 越界根因。

## 验证环境与口径

- 基线提交：`291bd7437bd4a91e95a0e9dba71aff5c1669be65`；远端独立 worktree：
  `/home/yjk/workspace/hyper-parallel-megamoe-review-20261104`，Git HEAD 为基线，候选修改由文件 SHA 清单标识。
- CANN：`9.1.0-beta.3`；独立环境：`hp-megamoe-cann91`，不修改已有 conda 环境。
- 硬件：Ascend 910B3，每卡 20 Cube / 40 Vector；EP=2/4、TP=1。
- 小形状：`T=128,H=5120,I_r=128,I_s=1792,E=4,K=2`。
  random 为每 token 随机 Top-K；zero_experts 将全部 token 发往专家 0/1；skewed 仅最后 16 个 token 改发其他专家。
- 正式形状：`T=4096,H=5120,I_r=1792,I_s=1792,E=32,K=8`、EP=4、BF16、EIO、静态 push。
  数值测试采用平衡路由和固定随机种子；性能使用既有固定形状 benchmark 的平衡路由、固定数据和权重。
- 数值参考：legacy 路由 MegaMoe + 外部共享分支，逐项比较输出、dX、dProbs、路由 W1/W2 梯度、共享 W1/W2 梯度。
  外部误差阈值保持 `rtol=0.05,atol=0.01`；保留图重复反向与反序 pending backward 均为 `rtol=0,atol=0`。
  对零 token 的本地专家，额外要求其两组路由权重梯度严格为零。

## 正确性结果

| 场景 | 配置与判据 | 结果 |
| --- | --- | --- |
| 主机侧图、ABI、绑定、复用检查 | `tests/ut/core/multicore` | 296 passed、1696 subtests passed |
| 冷进程小随机形状 | EP2，先外部对照再全融合 | 两个 rank 通过，无预热规避 |
| 首次配置顺序反转 | EP2，全融合先执行，再执行外部对照 | 两个 rank 通过 |
| 零接收 | EP2 的 rank 1、EP4 的 rank 2/3 无路由 token | 全输出/梯度通过，空专家梯度严格为零 |
| 严重偏斜 | EP2，最后 16 个 token 发往另一组专家 | 两个 rank 通过 |
| 随机路由全部开关组合 | EP2，64 组合 | 128 个 rank/config 检查通过 |
| 空接收全部开关组合 | EP2，64 组合，全新解压目录 | 128 个 rank/config 检查通过 |
| 正式形状 | EP4，外部对照与全融合 | 四个 rank 通过，取消共享梯度豁免 |
| 正式形状共享独立开关 | EP4，legacy 路由，sf/sb 四种组合 | 16 个 rank/config 检查通过 |
| 保留图与存储生命周期 | 全融合：首次反向后再反向 3 次、两个 pending forward 反序 backward、另一次 forward 不改旧输出 | 全部逐位比较通过 |
| FP16 支持范围探测 | EP2，现有公共接口 | 在 kernel 启动前由 BF16-only 校验拒绝，不计为数值通过 |
| 全新解压目录 | 全融合先执行、EP4 空接收、EP4 正式形状 | 已通过；正式形状经新 pytest 入口运行，1 passed |

早期空接收矩阵使用临时脚本的 240 秒上限，在仍正常推进时到时终止；没有将它计为通过。
最终归档回归使用 420 秒时限并从第一个组合重新运行，64 组合在时限内完成。
数值通过只针对上述形状、路由和硬件；不等于任意 tail、动态容量、TP、pull 或其他 SoC 均已验收。

### 正式形状逐边界误差

下表是四个 rank 中最大的 max-abs，RMS 列给出各 rank 的范围，比较对象始终是外部共享分支。
`bad` 是违反 `abs(actual-reference) <= 0.01 + 0.05*abs(reference)` 的元素数。

| 边界 | 修复前 max-abs | 修复后 max-abs | 修复前 RMS | 修复后 RMS | 修复后 bad / rank |
| --- | ---: | ---: | --- | --- | --- |
| shared up | 0 | 0 | 0 | 0 | 0 |
| shared activation | 0.0625 | 0 | 0.001717–0.001723 | 0 | 0 |
| shared dActivation | 0 | 0 | 0 | 0 | 0 |
| shared dUp | 0.001953125 | 0.000030518 | 0.00004077–0.00004080 | 1.33e-8–1.65e-8 | 0 |
| shared dW1 | 0.03125 | 0.015625 | 0.003290–0.003293 | 0.00001577–0.00002009 | 0 / 18,350,080 |
| shared dW2 | 0.015625 | 0 | 0.001931–0.001936 | 0 | 0 / 9,175,040 |

修复后 activation、dActivation、dW2 逐位一致；dW1 RMS 下降约 164–209 倍。
dUp 和 dW1 仍存在少量低精度元素差异，不能宣称所有梯度逐位一致；但原先 dW1 每 rank 的 31–43 个超阈值元素已降为零。
外部精度判断没有放宽，重复反向的逐位判据也没有放宽。

## 性能回归口径与结果

本轮针对上一轮最快配置和全融合做回归，**没有重新测量全部 64 组合的性能排名**。
两套 64 组合测试验证的是正确性。完整图见[上一轮算子图与测量口径](mega_moe_full_fusion_results_20261103.md#性能评测口径)。
本轮正式形状、路由公式、输入/权重种子、容量和计时范围与其一致：
`ids[t,k]=(8*t+k+8*rank)%32`，概率 `[8,7,6,5,4,3,2,1]/36`；
数据种子 `30000+rank`，两条路径构造参数前均重置为 `99173+rank`。
对照是同一 vendor 的 `legacy/legacy/sf0/sb0`，已包含路由主体 MegaKernel，并执行图外共享专家。

候选图为：

- 共享前向关、反向开：外部 shared Linear/SiLU/mul/Linear；路由 MegaKernel 的 unpermute 合入其结果；反向路由、共享及总 dX 在图内完成。
- 全融合：Gather/Dispatch → routed Up/SwiGLU/Down → Combine → shared Up/SwiGLU/Down → weighted unpermute 合并；反向共享分支复用完成后的路由临时区，图内合并 dX。
- 本轮只补归约上下文与共享舍入，task/event 数量、矩阵乘法规模、HBM 张量存活边界均保持上一轮设计。

每种配置启动两个独立进程组，backend 执行顺序正反各一次；每路径 1 次有效性检查、10 次预热、50 次正式采样。
计时包括路由准备、forward/backward 与框架调度，使用同步主机计时并同时记录 NPU Event；每次迭代先取 rank 最大值，再对两轮共 100 个样本汇总。
benchmark 检查梯度存在且有限，逐项数值精度另由前述 ST 验证，不能用性能脚本的有效性检查替代数值测试。
设备为 4–7，2 秒采样进程占用；检测到外部任务的三轮数据已废弃，最终四轮采样均未检测到其他任务。

Torch 峰值从首次有效性检查前重置，覆盖检查、预热和采样，包含已存活的参数/输入。
峰值取两轮较大值；节省量取相对同轮参考的两组差值中较小值。
物理 HBM 为迭代同步后采样最大值，不是连续监控的瞬时峰值；SHMEM 堆另记为 802 MiB。

| 配置 | F+B 中位数 ms | 同轮参考 ms | 耗时降幅 | Torch 分配峰值 MiB | 最少节省 MiB | 缓存池峰值 MiB | HBM 同步采样最大 MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `both/both/sf0/sb1` | 33.403 | 34.665 | 3.64% | 2435.10 | 7.12 | 2538 | 4298.12 |
| `both/both/sf1/sb1` | 34.392 | 34.694 | 0.87% | 2435.18 | 7.05 | 2518 | 4279.94 |

前者 F+B 均值为 33.558 ms，min/max 为 33.108/37.221 ms，NPU Event 中位数 33.254 ms；
后者均值为 34.478 ms，min/max 为 34.120/37.051 ms，Event 中位数 34.238 ms。
两者缓存池至少分别节省 332 / 352 MiB，物理 HBM 同步采样至少分别节省 361.61 / 364.32 MiB。
这些不同口径不能相加，也不能把分配峰值只下降约 7 MiB 解读为没有缓存池或物理占用收益。

| 配置 | 前向 AIC / AIV | 反向 AIC / AIV |
| --- | ---: | ---: |
| 外置参考 | 320 / 3073 | 640 / 3073 |
| `both/both/sf0/sb1` | 320 / 3329 | 720 / 3777 |
| `both/both/sf1/sb1` | 360 / 3361 | 720 / 3777 |

固定形状仍优先推荐 `both/both/False/True`。本轮不能重新确认所有 64 组合的全局最快或最省显存配置，也不改全局默认值。
相较上一轮，这两个候选的分配峰值没有增加；与早先缓冲区复用修复前相比，仍保留约 416 MiB 的下降。
跨日期的约 0.15 ms 变化不是受控的单项成本测量，不将其全部归因于舍入修复。

### 收益仍有限的原因

对照已经融合了路由主体；新增整合没有减少约 6.09 TFLOP/rank 的 F+B 主体矩阵乘法，也没有减少 SHMEM 通信。
全融合共享前向仍排在路由 Combine 后面，无法自动与路由计算重叠；共享反向也要等候路由读取者结束后才能复用存储。
本轮保留外部舍入语义需要 UB 内额外 cast/barrier，但没有增加 HBM 中间张量。
消除图外相加和避免额外分配的收益，必须扣除路由端点、更多任务、舍入及共享尾部调度的成本。
因此，统一到一次 launch 并不保证完整融合比部分融合更快。
已有内部追踪证明 dX 相加与共享 W1 梯度重叠，但不能据此声称共享前向也已重叠；
现有证据不足以将剩余差值精确分解成 GMM tiling、Vector 效率或每条 barrier 的占比。

## 构建与交付证据

远端结果目录：`/home/yjk/workspace/results/megamoe-review-20261104`。
使用既有隔离环境与标准 source build：

```bash
source /home/yjk/Ascend-9.1.0-beta.3/cann/set_env.sh
export PATH=/home/yjk/.conda/envs/hp-megamoe-cann91/bin:$PATH
export CPATH=$ASCEND_HOME_PATH/aarch64-linux/pkg_inc/base${CPATH:+:$CPATH}
bash hyper_parallel/core/multicore/build.sh --soc-list ascend910b --jobs 8
```

构建覆盖六个 native op、SHMEM 和 Torch adapter。验证六个配置的 `binList` 非空、metadata 的对象路径存在、对象 SHA 匹配；
MegaMoe 前向 arity 为 28/6、反向为 32/8，两个 dtype 的 `opParaSize=2512`。
核对 28 份组装/编译副本，以及 46 个 payload 文件的 SHA256。
交付前将补丁规范为无空白上下文的格式；重新组装后的 549 个源码文件与实际编译输入逐文件一致，原生制品无需变化。
最终归档 `native-payload-cann91-910b.tar.gz` 的 SHA256 为：
`73f521fb73f42a76f355d721ca2a2a63887e876c9c75195cd7f0c0ec536ec19a`。
当前仓库用 source build 交付，归档与明细保留在远端结果目录，不将旧格式 prebuild 混入源码。

[本轮证据 JSON](mega_moe_integration_review_evidence_20261104.json)包含修改后的源码 SHA、payload 身份、
四个 rank 的修复前后逐边界误差、正确性日志摘要、四轮性能的全部样本与进程占用结论。
原始日志还保留旧版失败、首次同步定位和被废弃的性能轮次。
本轮四个修改过的 Python 文件通过 Pylint；生产代码按既有 C19/L100 检查通过 46 个文件。
仓库的更严格 C15 门限仍会报告基线已有函数的复杂度，本轮没有新增该类告警；新增 ST 辅助函数通过 C15/L100。

## 后续改进建议

1. **优先扩大独立数值验收矩阵。** 已增加冷进程、首次配置顺序反转、空接收和偏斜路由的 pytest ST 入口，建议在设备 CI 固定执行；
   先扩展不同 shared width、hidden/intermediate tail 和实际训练数据；若计划支持 FP16，需先定义公共 dtype 契约。保留逐边界误差，以便区分数学舍入和 GMM 归约误差。
2. **优化共享前向调度。** 当前全融合减少了两次图外相加，但 shared forward 仍在 routed 终端之后串行执行。
   应先测量将共享 up projection 提前到通信阶段、或为共享分支保留少量 Cube 的收益，再决定是否采用。
   必须同时验证静态 worker 队列不会让等待任务占满 producer 所需资源；图无环并不足以证明不会死锁。
3. **减少长期维护中的隐式 ABI 假设。** 将指针槽位、2016-byte GMM tiling stride、SwiGLU tiling 字段定位集中定义，
   增加构建时布局校验和产物身份核对。当前 patch 已锁定来源版本和 SHA，更新 CANN 源码时必须重做验证。
4. **默认值继续按收益实测决定。** 全融合提供统一接口并不意味着共享前向一定更快。
   应分别考虑实际训练形状的延迟、Torch allocated/reserved、SHMEM 和物理 HBM；
   完整训练 step、优化器、checkpoint 与其他层重叠的峰值需要模型级验收，不能由本算子固定形状外推。
