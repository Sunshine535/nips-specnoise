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

# Stage 3: Noise-Guided SFT (4 strategies × 4 domains × 2 seeds) — one GPU per job
if ! is_phase_done 3; then
    echo "[Stage 3/5] Noise-Guided SFT (full ${NUM_GPUS}-GPU parallel)"

    JOBS=()
    for SEED in "${SEEDS[@]}"; do
        for DOMAIN in "${DOMAINS[@]}"; do
            for STRAT in "${STRATEGIES[@]}"; do
                JOBS+=("${SEED}:${DOMAIN}:${STRAT}")
            done
        done
    done
    echo "  Total SFT jobs: ${#JOBS[@]}, GPUs: ${NUM_GPUS}"

    for ((batch_start=0; batch_start < ${#JOBS[@]}; batch_start+=NUM_GPUS)); do
        PIDS=()
        FAIL3=0
        batch_end=$((batch_start + NUM_GPUS))
        [ "$batch_end" -gt "${#JOBS[@]}" ] && batch_end=${#JOBS[@]}

        for ((i=batch_start; i < batch_end; i++)); do
            IFS=':' read -r SEED DOMAIN STRAT <<< "${JOBS[$i]}"
            G=$(( i - batch_start ))
            PHYS_G=$(gpu_at_index $G)
            echo "  [$((i+1))/${#JOBS[@]}] GPU ${PHYS_G}: seed=${SEED} domain=${DOMAIN} strategy=${STRAT}"
            CUDA_VISIBLE_DEVICES=$PHYS_G python "${SCRIPT_DIR}/run_noise_guided_sft.py" \
                --config_path "${CONFIG}" --noise_results "${SEARCH_RESULTS}" \
                --output_dir "${RESULTS}/noise_guided_sft" \
                --domains "$DOMAIN" --strategies "$STRAT" \
                --seed "$SEED" \
                > "${LOG_DIR}/stage3_sft_${DOMAIN}_${STRAT}_seed${SEED}.log" 2>&1 &
            PIDS+=($!)
        done

        for pid in "${PIDS[@]}"; do
            wait "$pid" || FAIL3=1
        done
        [ "$FAIL3" -eq 0 ] || { echo "  [ERROR] Batch starting at $((batch_start+1)) failed. Check logs."; exit 1; }
    done
    phase_done 3
fi

# Stage 4 + Stage 5: domain eval (one GPU per domain) + 27B Fisher validation (remaining GPUs)
if ! is_phase_done 4 || ! is_phase_done 5; then
    echo "[Stages 4+5] Domain evaluation + 27B validation (parallel)"
    P4_PIDS=()
    GPU_IDX=0

    if ! is_phase_done 4; then
        for DOMAIN in "${DOMAINS[@]}"; do
            G=$((GPU_IDX % NUM_GPUS))
            PHYS_G=$(gpu_at_index $G)
            echo "  Evaluating domain: ${DOMAIN} on GPU ${PHYS_G}"
            MODEL_PATHS="baseline:${MODEL_BASE}"
            for STRAT in "${STRATEGIES[@]}"; do
                CKPT="${RESULTS}/noise_guided_sft/${DOMAIN}/${STRAT}/seed42"
                if [ -d "$CKPT" ]; then
                    MODEL_PATHS="${MODEL_PATHS} ${STRAT}:${CKPT}"
                fi
            done
            CUDA_VISIBLE_DEVICES=$PHYS_G python "${SCRIPT_DIR}/eval_domain_performance.py" \
                --config_path "${CONFIG}" --model_paths ${MODEL_PATHS} \
                --output_dir "${RESULTS}/domain_eval/${DOMAIN}" \
                --base_model "${MODEL_BASE}" --domains "${DOMAIN}" mmlu --max_samples 200 \
                > "${LOG_DIR}/stage4_eval_${DOMAIN}.log" 2>&1 &
            P4_PIDS+=($!)
            GPU_IDX=$((GPU_IDX + 1))
        done
    fi

    P5_PID=""
    if ! is_phase_done 5; then
        REMAINING_GPUS=""
        for ((g=GPU_IDX; g<NUM_GPUS; g++)); do
            local_g=$(gpu_at_index $g)
            [ -n "$REMAINING_GPUS" ] && REMAINING_GPUS="${REMAINING_GPUS},"
            REMAINING_GPUS="${REMAINING_GPUS}${local_g}"
        done
        [ -z "$REMAINING_GPUS" ] && REMAINING_GPUS="$(gpu_at_index $((NUM_GPUS - 1)))"
        echo "  27B validation on GPUs ${REMAINING_GPUS}"
        CUDA_VISIBLE_DEVICES=$REMAINING_GPUS python "${SCRIPT_DIR}/run_fisher_analysis.py" \
            --config_path "${CONFIG}" --output_dir "${RESULTS}/fisher_analysis_27b" \
            --noise_type gaussian --noise_scale 0.01 --num_samples 100 \
            > "${LOG_DIR}/stage5_27b.log" 2>&1 &
        P5_PID=$!
    fi

    FAIL4=0
    for pid in "${P4_PIDS[@]}"; do
        wait "$pid" || FAIL4=1
    done
    if [ -n "$P5_PID" ]; then
        wait "$P5_PID" || echo "  27B validation skipped (insufficient memory)"
    fi
    [ "$FAIL4" -eq 0 ] || exit 1

    if ! is_phase_done 4; then phase_done 4; fi
    if ! is_phase_done 5; then phase_done 5; fi
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
