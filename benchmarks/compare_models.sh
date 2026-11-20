#!/usr/bin/env bash
# The benchmark behind the README's charts and tables: Nenya-2.5B-A0.5B, Qwen3-0.6B and Qwen3.5-2B on one GPU, with the
# same vLLM (0.33.0) and the same settings for every model. Each run is offline (benchmarks/compare_offline.py), then
# online (benchmarks/compare_serve.sh). Two rounds in alternating order (Nenya first, then last), so slow drift on the
# box (clocks, temperature) does not favour one model. About 25 minutes per model and round on an RTX 5090.
#
#   WITH_QWEN=1 bash benchmarks/setup_box.sh       # once: environment, checkpoints
#   benchmarks/compare_models.sh                   # writes results/<model>_r<round>/
#   python benchmarks/summarize.py                 # the tables, mean of the rounds
#
# Environment (all optional): MODELS (default "nenya qwen3-0.6b qwen3.5-2b"), ROUNDS (default 2), MODELS_DIR (default
# /root/models). The Qwen3.5 checkpoint is a vision-language model: its vision encoder is switched off (0 images and
# videos per prompt), so every model is compared as a text model.
set -uo pipefail
cd "$(dirname "$0")/.."
MODELS_DIR=${MODELS_DIR:-/root/models}
NO_MM='{"limit_mm_per_prompt": {"image": 0, "video": 0}}'
# Triton and fla keep their autotuning results on disk, so every run of a model uses the same kernel configurations
# (as in the published runs).
export TRITON_CACHE_AUTOTUNING=1 FLA_CACHE_RESULTS=1

run_model() {   # run_model NAME MODEL_DIR OFFLINE_EXTRA_JSON [extra vllm serve args]
    local name=$1 model=$2 extra=$3
    shift 3
    mkdir -p "results/$name"
    local t0=$SECONDS
    python benchmarks/compare_offline.py --model "$model" --name "$name" --extra "$extra" \
        > "results/$name/offline.log" 2>&1
    echo "$name offline: exit $? ($((SECONDS - t0)) s)"
    t0=$SECONDS
    bash benchmarks/compare_serve.sh "$model" "$name" "$@"
    echo "$name serve: ($((SECONDS - t0)) s)"
}

run() {   # run MODEL ROUND
    case $1 in
        nenya)      run_model "nenya_r$2" "$MODELS_DIR/nenya-base" '{}' ;;
        qwen3-0.6b) run_model "qwen3-0.6b_r$2" "$MODELS_DIR/qwen3-0.6b-base" '{}' ;;
        qwen3.5-2b) run_model "qwen3.5-2b_r$2" "$MODELS_DIR/qwen3.5-2b-base" "$NO_MM" \
                        --limit-mm-per-prompt '{"image": 0, "video": 0}' ;;
        *) echo "unknown model: $1 (nenya, qwen3-0.6b or qwen3.5-2b)"; return 1 ;;
    esac
}

read -r -a models <<< "${MODELS:-nenya qwen3-0.6b qwen3.5-2b}"
for r in $(seq 1 "${ROUNDS:-2}"); do
    order=("${models[@]}")
    if [ $((r % 2)) = 0 ]; then   # even rounds run the models in reverse order
        order=()
        for ((i = ${#models[@]} - 1; i >= 0; i--)); do order+=("${models[$i]}"); done
    fi
    for m in "${order[@]}"; do run "$m" "$r"; done
done
echo COMPARE_DONE
