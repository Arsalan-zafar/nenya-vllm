#!/usr/bin/env bash
# Online comparison: one model behind `vllm serve` with the settings shared by every model compared, driven through
# /v1/completions by `vllm bench serve` (random token prompts of the model's own tokenizer, ignore_eos, greedy).
#
#   benchmarks/compare_serve.sh /root/models/qwen3.5-2b-base qwen3.5-2b [extra vllm serve args]
#
# Writes results/<name>/serve/<scenario>.json|log, server.log, nvsmi.csv (power/utilisation every 200 ms)
# and scenarios.tsv (start/end time of each scenario, to slice nvsmi.csv).
#   chat_c{1,16,64,256}   1,024-token prompt, 256 output tokens, closed loop at 1/16/64/256 concurrent requests
#   long_c{1,16}          8,192-token prompt, 256 output tokens: prefill-heavy, at 1 and 16 concurrent requests
#   tools_c32             4 distinct 4,096-token prefixes shared by all requests (a tool-calling server's system
#                         prompt + schemas), 256-token suffix, 128 output tokens, 32 concurrent; prefix caching on
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:?model dir}
NAME=${2:?name}
shift 2
OUT=results/$NAME/serve
PORT=${PORT:-8000}
mkdir -p "$OUT"
: > "$OUT/scenarios.tsv"

nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,utilization.memory,memory.used,clocks.sm \
    --format=csv,noheader,nounits -lms 200 > "$OUT/nvsmi.csv" &
SMI_PID=$!

t0=$(date +%s.%N)
vllm serve "$MODEL" --served-model-name m --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.85 \
    --max-num-batched-tokens 8192 --port "$PORT" "$@" > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
ready=0
for _ in $(seq 1 240); do
    if curl -sf "http://localhost:$PORT/health" > /dev/null; then ready=1; break; fi
    kill -0 "$SERVER_PID" 2> /dev/null || break
    sleep 2
done
t1=$(date +%s.%N)
echo "startup_s $(python3 -c "print(round($t1 - $t0, 1))")" > "$OUT/startup.txt"
if [ "$ready" != 1 ]; then
    echo "server did not start"; tail -30 "$OUT/server.log"; kill "$SMI_PID"; exit 1
fi

bench() {   # bench NAME ARGS...
    local name=$1
    shift
    local a b
    a=$(date +%s.%N)
    vllm bench serve --backend vllm --endpoint /v1/completions --port "$PORT" --model m --tokenizer "$MODEL" \
        --trust-remote-code --ignore-eos --seed 0 --temperature 0 \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --save-result --result-dir "$OUT" --result-filename "$name.json" "$@" > "$OUT/$name.log" 2>&1
    local rc=$?
    b=$(date +%s.%N)
    printf "%s\t%s\t%s\n" "$name" "$a" "$b" >> "$OUT/scenarios.tsv"
    echo "  $NAME $name: exit $rc"
}

for c in 1 16 64 256; do
    n=$((c == 1 ? 32 : (c < 64 ? 128 : 2 * c)))
    bench "chat_c$c" --dataset-name random --random-input-len 1024 --random-output-len 256 --random-range-ratio 0 \
        --num-prompts "$n" --max-concurrency "$c"
done
bench long_c1 --dataset-name random --random-input-len 8192 --random-output-len 256 --random-range-ratio 0 \
    --num-prompts 12 --max-concurrency 1
bench long_c16 --dataset-name random --random-input-len 8192 --random-output-len 256 --random-range-ratio 0 \
    --num-prompts 64 --max-concurrency 16
bench tools_c32 --dataset-name prefix_repetition --prefix-repetition-prefix-len 4096 \
    --prefix-repetition-suffix-len 256 --prefix-repetition-num-prefixes 4 --prefix-repetition-output-len 128 \
    --num-prompts 256 --max-concurrency 32

kill "$SERVER_PID" 2> /dev/null
wait "$SERVER_PID" 2> /dev/null
kill "$SMI_PID" 2> /dev/null
echo "  $NAME serve done"
