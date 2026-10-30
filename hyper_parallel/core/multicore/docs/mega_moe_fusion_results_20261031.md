# MegaMoe Fusion Results, 2026-10-31

> Superseded by the [complete, controlled 64-combination report](mega_moe_switch_matrix_results_20261102.md).
> The historical common/MegaMoe comparison below did not perform equal work:
> MegaMoe implicitly included an external shared expert while common did not.
> Historical shared references also inherited each candidate's route modes.
> Retain these numbers as historical observations only; do not use their ratios
> to select defaults or claim an equal-work speedup.

The route and shared-expert switches were measured on the existing fixed MegaMoe benchmark shape: 4 ranks / EP4, 4096 local tokens, hidden 5120, routed intermediate 1792, 32 experts, top-k 8, BF16. The host was Ascend 910B3 with CANN 9.1 beta.3, Torch 2.7.1 and torch_npu 2.7.1.post4. Each formal comparison used 8 warmup and 30 measured forward/backward iterations. Latency is the maximum rank time; gradients were required to be present and finite.

| Configuration | Reference median | Candidate median | Change | Reference peak device used | Candidate peak device used | SHMEM heap | Peak Torch allocated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Route legacy vs common | 61.80 ms | 34.53 ms | -44.1% | 4.87 GiB | 5.02 GiB | 0.78 GiB | 2.39 GiB |
| Route forward permute vs common | 61.83 ms | 34.76 ms | -43.8% | 4.87 GiB | 5.06 GiB | 0.78 GiB | 2.39 GiB |
| Route forward unpermute vs common | 61.69 ms | 34.56 ms | -44.0% | 4.87 GiB | 5.08 GiB | 0.78 GiB | 2.39 GiB |
| Route backward permute vs common | 62.03 ms | 34.52 ms | -44.3% | 4.87 GiB | 5.04 GiB | 0.78 GiB | 2.78 GiB |
| Route backward unpermute vs common | 61.87 ms | 34.48 ms | -44.3% | 4.87 GiB | 5.08 GiB | 0.78 GiB | 2.78 GiB |
| Shared forward fused vs external shared | 34.67 ms | 36.07 ms | +4.0% | 4.47 GiB | 4.44 GiB | 0.78 GiB | 2.34 GiB |
| Shared backward fused vs external shared | 34.63 ms | 34.15 ms | -1.4% | 4.47 GiB | 4.78 GiB | 0.78 GiB | 2.78 GiB |
| Shared forward and backward fused vs external shared | 34.61 ms | 35.06 ms | +1.3% | 4.47 GiB | 4.79 GiB | 0.78 GiB | 2.78 GiB |

Route endpoint variants are within 0.2% in latency. The sampled physical-device peak for route fusion is about 0.02-0.06 GiB above the legacy run and about 0.17-0.21 GiB above common. These measurements do not justify enabling route fusion by default. The defaults are therefore forward route `legacy`, backward route `legacy`, shared forward disabled, and shared backward disabled. Every direction remains independently selectable through `forward_fusion_mode`, `backward_fusion_mode`, `forward_shared_expert`, and `backward_shared_expert`.

Shared backward saves about 0.48 ms here, at the cost of about 0.31 GiB more sampled physical-device peak and about 406 MiB more peak Torch allocated memory. Shared forward adds about 1.40 ms while reducing the sampled peak by about 0.03 GiB. Enabling both adds about 0.45 ms and about 0.32 GiB. These are single-layer measurements; they do not model activation retention across a deep training graph. Shared tasks are conservatively ordered after the routed branch in the current schedule, so these results should not be interpreted as evidence for overlap.

Peak device used is sampled after synchronization at each warmup/measured iteration boundary using `torch.npu.mem_get_info`; it includes allocator reservations, SHMEM, and other device allocations visible to the driver. It is a sampled peak, not a continuous high-water mark. Peak Torch allocated/reserved and configured SHMEM heap are reported separately in the accompanying [CSV](benchmarks/mega_moe_fusion_20261031.csv). The common baseline samples about 4.87 GiB device use. Shared comparisons use the same route mode on their external and fused legs. Full samples and exact byte counts are retained in the worker JSON under `/tmp/pytest-of-yjk/` on the benchmark host.
