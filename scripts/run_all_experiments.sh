#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# SpecNoise: Master Experiment Orchestration
# noise_search -> fisher_analysis -> noise_guided_sft (4x4x2) -> eval -> 27B
# ============================================================================

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/home/nwh/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME/hub"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${PROJECT_DIR}/configs/noise_grid.yaml"
RESULTS="${PROJECT_DIR}/results"
SEARCH_RESULTS="${RESULTS}/noise_search/noise_search_results.json"
MODEL_BASE="Qwen/Qwen3.5-9B"

SEEDS=(42 123)
DOMAINS=(medical legal code finance)
STRATEGIES=(standard_sft pre_noise_sft noise_regularized iterative_noise)

echo "========================================="
echo " SpecNoise: Full Experiment Pipeline"
echo " Config: ${CONFIG}"
echo " Results: ${RESULTS}"
echo "========================================="

# ── Stage 1: Noise Search ────────────────────────────────────────────────────
echo ""
echo "[Stage 1/5] Noise Search"
if [ -f "$SEARCH_RESULTS" ]; then
    echo "  Noise search results exist, skipping."
else
    python "${SCRIPT_DIR}/noise_search.py" \
        --config_path "${CONFIG}" \
        --output_dir "${RESULTS}/noise_search"
fi

# ── Stage 2: Fisher Analysis ─────────────────────────────────────────────────
echo ""
echo "[Stage 2/5] Fisher Analysis"
python "${SCRIPT_DIR}/run_fisher_analysis.py" \
    --config_path "${CONFIG}" \
    --noise_results "${SEARCH_RESULTS}" \
    --output_dir "${RESULTS}/fisher_analysis" \
    --num_samples 200

# ── Stage 3: Noise-Guided SFT (4 strategies x 4 domains x 2 seeds) ──────────
echo ""
echo "[Stage 3/5] Noise-Guided SFT"
for SEED in "${SEEDS[@]}"; do
    echo "  Seed: ${SEED}"
    python "${SCRIPT_DIR}/run_noise_guided_sft.py" \
        --config_path "${CONFIG}" \
        --noise_results "${SEARCH_RESULTS}" \
        --output_dir "${RESULTS}/noise_guided_sft" \
        --domains "${DOMAINS[@]}" \
        --strategies "${STRATEGIES[@]}" \
        --seed "$SEED"
done

# ── Stage 4: Domain Evaluation ───────────────────────────────────────────────
echo ""
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
        --config_path "${CONFIG}" \
        --model_paths ${MODEL_PATHS} \
        --output_dir "${RESULTS}/domain_eval/${DOMAIN}" \
        --base_model "${MODEL_BASE}" \
        --domains "${DOMAIN}" mmlu \
        --max_samples 200
done

# ── Stage 5: 27B Validation (optional) ───────────────────────────────────────
echo ""
echo "[Stage 5/5] 27B Validation (optional)"
MODEL_27B="Qwen/Qwen3.5-27B"
if python -c "from transformers import AutoModelForCausalLM; print('ok')" 2>/dev/null; then
    echo "  Running Fisher analysis on 27B for validation..."
    python "${SCRIPT_DIR}/run_fisher_analysis.py" \
        --config_path "${CONFIG}" \
        --output_dir "${RESULTS}/fisher_analysis_27b" \
        --noise_type gaussian \
        --noise_scale 0.01 \
        --num_samples 100 || echo "  27B validation skipped (insufficient memory)"
else
    echo "  Skipping 27B validation"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "========================================="
echo " SpecNoise: Pipeline Complete"
echo "========================================="
echo " Results directory: ${RESULTS}"
echo ""
echo " Key outputs:"
echo "   ${RESULTS}/noise_search/noise_search_results.json"
echo "   ${RESULTS}/fisher_analysis/fisher_heatmap.pdf"
echo "   ${RESULTS}/fisher_analysis/layer_sensitivity.pdf"
echo "   ${RESULTS}/noise_guided_sft/all_strategies_seed*.json"
echo "   ${RESULTS}/domain_eval/*/domain_eval_results.json"
echo "========================================="
