#!/usr/bin/env bash
set -euo pipefail

# run_pipeline_fix.sh
# ──────────────────────────────────────────────────────────────────────────────
# Patches the analysis after the initial run_pipeline.sh, in two parts:
#
#   A) Fix the multiview probe selection.
#      run_pipeline.sh:54 trained `multiview` with --probes phase,denoiser, but
#      results/invariants/ranking.md recommends bicoherence+denoiser as the
#      top-2 probes (phase is the WORST probe — OOD AUC 0.409, below chance).
#      We archive the buggy runs, re-train with the correct probes across all
#      5 seeds, and re-compute significance.
#
#   B) Add new architectural baselines.
#      ConvNeXt, ViT, EfficientViT (general vision backbones on log-mel input)
#      plus Specttra-{alpha,beta,gamma} from SONICS
#      (https://awsaf49.github.io/sonics-website/). Trained no-aug, seed 0,
#      matching the baseline protocol of the original pipeline.
#
# ⚠️  PREREQUISITE for section B
#      scripts/training/train_baseline.py currently registers only:
#        lcnn, mert_head, muq_head, moss_nano_head, clap_head, multiview
#      Before section B will run, that file must be extended to register the
#      new --model choices:
#        convnext, vit, efficientvit, specttra_alpha, specttra_beta, specttra_gamma
#      (add to the argparse `choices=[...]` list and add corresponding model
#      construction branches in run_training()). Until then, section B will
#      exit with an argparse error and section C will fail to find the runs.
#
# Re-runnability: each step is guarded by a .done stamp in results/pipeline_state/
# (same convention as run_pipeline.sh). Set FORCE_PIPELINE_RUN=1 to re-execute.
# ──────────────────────────────────────────────────────────────────────────────

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

# ── A. Fix the multiview probe selection ─────────────────────────────────────

# A.1 — Move the buggy (phase,denoiser) multiview artifacts out of the way so
#       the retrain doesn't overwrite them silently. summary.csv is left alone
#       so the buggy rows remain in the audit trail.
run_step "10b_archive_buggy_multiview" "archive multiview runs trained with --probes phase,denoiser" \
  bash -c '
    set -e
    mkdir -p checkpoints/_archive_phase_denoiser
    mkdir -p results/scores/_archive_phase_denoiser
    for s in 0 1 2 3 4; do
      ckpt="checkpoints/multiview_aug-combined_safe_seed${s}"
      if [[ -d "$ckpt" ]]; then
        mv "$ckpt" "checkpoints/_archive_phase_denoiser/"
      fi
      for split in val test ood; do
        f="results/scores/multiview_aug-combined_safe_seed${s}_${split}.npz"
        if [[ -f "$f" ]]; then
          mv "$f" "results/scores/_archive_phase_denoiser/"
        fi
      done
    done
  '

# A.2 — Re-train multiview with the correct top-2 probes from ranking.md.
run_step "10c_train_multiview_correct_probes" "retrain multiview with bicoherence,denoiser (5 seeds)" \
  uv run scripts/training/train_baseline.py --model multiview \
    --probes bicoherence,denoiser --augment combined_safe --swa --tta --seeds 0,1,2,3,4

# A.3 — Re-run significance against the original 5 baselines. This overwrites
#       results/baselines/comparison.md; the corrected version is what the
#       paper should cite.
run_step "11b_stats_significance_corrected" "significance: corrected multiview vs. original baselines" \
  uv run scripts/training/stats_significance.py \
    --proposed multiview_aug-combined_safe_seed \
    --baselines lcnn_aug-none_seed0,mert_head_aug-none_seed0,muq_head_aug-none_seed0,moss_nano_head_aug-none_seed0,clap_head_aug-none_seed0

# ── B. New architectural baselines (REQUIRES train_baseline.py extension) ────
#
# All trained with the same protocol as the existing baselines: no augmentation,
# seed 0, default epochs. Add more seeds later if any of these become a headline
# number in the paper.

run_step "12_train_convnext"        "train ConvNeXt baseline (mel input)" \
  uv run scripts/training/train_baseline.py --model convnext

run_step "13_train_vit"             "train ViT baseline (mel input)" \
  uv run scripts/training/train_baseline.py --model vit

run_step "14_train_efficientvit"    "train EfficientViT baseline (mel input)" \
  uv run scripts/training/train_baseline.py --model efficientvit

run_step "15_train_specttra_alpha"  "train Specttra-alpha (SONICS)" \
  uv run scripts/training/train_baseline.py --model specttra_alpha

run_step "16_train_specttra_beta"   "train Specttra-beta (SONICS)" \
  uv run scripts/training/train_baseline.py --model specttra_beta

run_step "17_train_specttra_gamma"  "train Specttra-gamma (SONICS)" \
  uv run scripts/training/train_baseline.py --model specttra_gamma

# ── C. Final significance comparison including new baselines ─────────────────
# This overwrites results/baselines/comparison.md again; it is the final
# artifact for the paper and supersedes 11b's output.
run_step "18_stats_significance_full" "significance: corrected multiview vs. all baselines (original + new)" \
  uv run scripts/training/stats_significance.py \
    --proposed multiview_aug-combined_safe_seed \
    --baselines lcnn_aug-none_seed0,mert_head_aug-none_seed0,muq_head_aug-none_seed0,moss_nano_head_aug-none_seed0,clap_head_aug-none_seed0,convnext_aug-none_seed0,vit_aug-none_seed0,efficientvit_aug-none_seed0,specttra_alpha_aug-none_seed0,specttra_beta_aug-none_seed0,specttra_gamma_aug-none_seed0

echo ""
echo "Done. Updated artifacts:"
echo "  - results/baselines/comparison.md     (corrected multiview vs. all baselines)"
echo "  - results/scores/multiview_aug-combined_safe_seed{0..4}_{val,test,ood}.npz"
echo "  - results/scores/{convnext,vit,efficientvit,specttra_*}_aug-none_seed0_*.npz"
echo "  - checkpoints/_archive_phase_denoiser/  (preserved buggy multiview runs)"
echo ""
echo "Remember to regenerate summary.md from the updated results."
