#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu_utils.sh"
auto_setup

PROJ_DIR_ROOT="$(dirname "$SCRIPT_DIR")"
if [ -f "$PROJ_DIR_ROOT/.venv/bin/activate" ]; then
    source "$PROJ_DIR_ROOT/.venv/bin/activate"
fi
export PATH="$HOME/.local/bin:$PATH"

# --- Phase resume ---
PHASE_MARKER_DIR="$PROJ_DIR_ROOT/results/.phase_markers"
mkdir -p "$PHASE_MARKER_DIR"
FORCE_RERUN="${FORCE_RERUN:-0}"
phase_done() { touch "$PHASE_MARKER_DIR/phase_${1}.done"; echo "[PHASE $1] Completed at $(date)"; }
is_phase_done() {
    [[ "$FORCE_RERUN" == "1" ]] && return 1
    [[ -f "$PHASE_MARKER_DIR/phase_${1}.done" ]] && echo "[PHASE $1] Already completed. Skipping." && return 0
    return 1
}

PROJECT_DIR="$PROJ_DIR_ROOT"
CONFIG="${PROJECT_DIR}/configs/noise_grid.yaml"
RESULTS="${PROJECT_DIR}/results"
SEARCH_RESULTS="${RESULTS}/noise_search/noise_search_results.json"
MODEL_BASE="Qwen/Qwen3.5-9B"
LOG_DIR="${PROJECT_DIR}/logs"

SEEDS=(42 123)
DOMAINS=(medical legal code finance)
STRATEGIES=(standard_sft pre_noise_sft noise_regularized iterative_noise)

mkdir -p "$LOG_DIR"

echo "========================================="
echo " SpecNoise: Full Experiment Pipeline"
echo " GPUs: $NUM_GPUS × $GPU_CLASS"
echo "========================================="

# Stage 1: Noise Search
if ! is_phase_done 1; then
    echo "[Stage 1/5] Noise Search"
    python "${SCRIPT_DIR}/noise_search.py" \
        --config_path "${CONFIG}" --output_dir "${RESULTS}/noise_search" \
        2>&1 | tee "${LOG_DIR}/stage1_noise_search.log"
    phase_done 1
fi

# Stage 2: Fisher Analysis
if ! is_phase_done 2; then
    echo "[Stage 2/5] Fisher Analysis"
    python "${SCRIPT_DIR}/run_fisher_analysis.py" \
        --config_path "${CONFIG}" --noise_results "${SEARCH_RESULTS}" \
        --output_dir "${RESULTS}/fisher_analysis" --num_samples 200 \
        2>&1 | tee "${LOG_DIR}/stage2_fisher.log"
    phase_done 2
fi

# Stage 3: Noise-Guided SFT (4 strategies × 4 domains × 2 seeds)
if ! is_phase_done 3; then
    echo "[Stage 3/5] Noise-Guided SFT"
    for SEED in "${SEEDS[@]}"; do
        echo "  Seed: ${SEED}"
        python "${SCRIPT_DIR}/run_noise_guided_sft.py" \
            --config_path "${CONFIG}" --noise_results "${SEARCH_RESULTS}" \
            --output_dir "${RESULTS}/noise_guided_sft" \
            --domains "${DOMAINS[@]}" --strategies "${STRATEGIES[@]}" \
            --seed "$SEED" \
            2>&1 | tee "${LOG_DIR}/stage3_sft_seed${SEED}.log"
    done
    phase_done 3
fi

# Stage 4: Domain Evaluation
if ! is_phase_done 4; then
    echo "[Stage 4/5] Domain Evaluation"
    for DOMAIN in "${DOMAINS[@]}"; do
        echo "  Evaluating domain: ${DOMAIN}"
        MODEL_PATHS="baseline:${MODEL_BASE}"
        for STRAT in "${STRATEGIES[@]}"; do
            CKPT="${RESULTS}/noise_guided_sft/${DOMAIN}/${STRAT}/seed42"
            if [ -d "$CKPT" ]; then
                MODEL_PATHS="${MODEL_PATHS} ${STRAT}:${CKPT}"
            fi
        done
        python "${SCRIPT_DIR}/eval_domain_performance.py" \
            --config_path "${CONFIG}" --model_paths ${MODEL_PATHS} \
            --output_dir "${RESULTS}/domain_eval/${DOMAIN}" \
            --base_model "${MODEL_BASE}" --domains "${DOMAIN}" mmlu --max_samples 200 \
            2>&1 | tee "${LOG_DIR}/stage4_eval_${DOMAIN}.log"
    done
    phase_done 4
fi

# Stage 5: 27B Validation
if ! is_phase_done 5; then
    echo "[Stage 5/5] 27B Validation"
    python "${SCRIPT_DIR}/run_fisher_analysis.py" \
        --config_path "${CONFIG}" --output_dir "${RESULTS}/fisher_analysis_27b" \
        --noise_type gaussian --noise_scale 0.01 --num_samples 100 \
        2>&1 | tee "${LOG_DIR}/stage5_27b.log" || echo "  27B validation skipped (insufficient memory)"
    phase_done 5
fi

echo "========================================="
echo " SpecNoise: Pipeline Complete"
echo "========================================="

DONE_FILE="$PROJ_DIR_ROOT/results/.pipeline_done"
cat > "$DONE_FILE" << DONEEOF
{
  "project": "nips-specnoise",
  "completed_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "hostname": "$(hostname)",
  "gpus": "${NUM_GPUS:-unknown}",
  "status": "PIPELINE_COMPLETE"
}
DONEEOF
echo "[PIPELINE_COMPLETE] Run 'bash collect_results.sh' to package results."
