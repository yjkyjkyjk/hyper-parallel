#!/usr/bin/env bash
set -u

# Activate the isolated CANN/Torch environment and matching vendor before running.

root=${1:-/tmp/hp-mega-moe-switch-matrix}
mkdir -p "$root"
summary="$root/matrix.log"
printf 'START %s\n' "$(date -Is)" >> "$summary"

for f in legacy permute unpermute both; do
  for b in legacy permute unpermute both; do
    for sf in 0 1; do
      for sb in 0 1; do
        case_name="f${f}_b${b}_sf${sf}_sb${sb}"
        case_dir="$root/$case_name"
        result="$case_dir/performance/routed_expert_ab.json"
        if [[ -s "$result" ]]; then
          printf 'SKIP %s\n' "$case_name" >> "$summary"
          continue
        fi
        mkdir -p "$case_dir"
        printf 'START %s %s\n' "$(date -Is)" "$case_name" >> "$summary"
        env \
          HP_MEGA_MOE_PERF_WARMUP=8 \
          HP_MEGA_MOE_PERF_MEASURED=30 \
          HP_MEGA_MOE_MAX_PERF_RATIO=100 \
          HP_MEGA_MOE_EVIDENCE_DIR="$case_dir" \
          HP_MEGA_MOE_SHARED_INTERMEDIATE_SIZE=1792 \
          HP_MEGA_MOE_FORWARD_SHARED_EXPERT="$sf" \
          HP_MEGA_MOE_BACKWARD_SHARED_EXPERT="$sb" \
          HYPER_PARALLEL_MEGA_MOE_FORWARD_FUSION_MODE="$f" \
          HYPER_PARALLEL_MEGA_MOE_BACKWARD_FUSION_MODE="$b" \
          pytest -q tests/torch/multicore/test_mega_moe.py -k representative_performance --disable-warnings --tb=short \
          >"$case_dir/pytest.log" 2>&1
        rc=$?
        if [[ $rc -eq 0 && -s "$result" ]]; then
          printf 'DONE %s %s\n' "$(date -Is)" "$case_name" >> "$summary"
        else
          printf 'FAIL %s rc=%s\n' "$case_name" "$rc" >> "$summary"
        fi
      done
    done
  done
done
printf 'END %s\n' "$(date -Is)" >> "$summary"
