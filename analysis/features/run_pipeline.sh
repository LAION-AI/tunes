#!/usr/bin/env bash
set -euo pipefail

STATE_DIR="results/pipeline_state"
mkdir -p "$STATE_DIR"

run_step() {
  local step_id="$1"
  local description="$2"
  shift 2
  local done_file="$STATE_DIR/${step_id}.done"

  if [[ "${FORCE_PIPELINE_RUN:-0}" != "1" && -f "$done_file" ]]; then
    echo "[SKIP] ${step_id}: ${description} (already completed)"
    return
  fi

  echo "[RUN]  ${step_id}: ${description}"
  "$@"
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$done_file"
  echo "[DONE] ${step_id}: ${description}"
}

# Step 1 - Cache features for all songs in the dataset
run_step "01_cache_features" "cache handcrafted features" \
  uv run scripts/training/cache_features.py

# Step 2 (longest — ~few hours)
run_step "02_cache_foundation_embeddings" "cache foundation embeddings" \
  uv run scripts/training/cache_foundation_embeddings.py --models mert,muq,moss_nano,clap

# Step 3
run_step "03_probe_invariants" "probe handcrafted invariants" \
  uv run scripts/training/probe_invariants.py --all

# Steps 4/5
run_step "04_train_mert_head" "train mert head baseline" \
  uv run scripts/training/train_baseline.py --model mert_head
run_step "05_train_muq_head" "train muq head baseline" \
  uv run scripts/training/train_baseline.py --model muq_head
run_step "06_train_moss_nano_head" "train moss nano head baseline" \
  uv run scripts/training/train_baseline.py --model moss_nano_head
run_step "07_train_clap_head" "train clap head baseline" \
  uv run scripts/training/train_baseline.py --model clap_head
run_step "08_train_lcnn" "train lcnn baseline" \
  uv run scripts/training/train_baseline.py --model lcnn

# Step 6
run_step "09_ablate_augmentations" "run augmentation ablations" \
  uv run scripts/training/ablate_augmentations.py --regimes all

# Step 7 (read top-2 probes from results/invariants/ranking.md first)
run_step "10_train_multiview" "train multiview final model" \
  uv run scripts/training/train_baseline.py --model multiview --probes phase,denoiser --augment combined_safe --swa --tta --seeds 0,1,2,3,4

# Step 8
run_step "11_stats_significance" "compute significance statistics" \
  uv run scripts/training/stats_significance.py --proposed multiview_aug-combined_safe_seed --baselines lcnn_aug-none_seed0,mert_head_aug-none_seed0,muq_head_aug-none_seed0,moss_nano_head_aug-none_seed0,clap_head_aug-none_seed0