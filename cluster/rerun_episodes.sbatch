#!/bin/bash
#SBATCH -J prime-rerun
#SBATCH -A <account>
#SBATCH -N 1
#SBATCH --requeue
#SBATCH -o logs/%x-%j.out
# Rerun the episodes of results/rerun_manifest.csv with the corrected runner
# (budget charged by each call's own cache status). Serves MODEL with vLLM the
# way serve_and_run_guided.sbatch did for the logged runs (same caches on
# scratch, parser from the model id, eager mode, 16k context, greedy decoding),
# checks tool calling, and runs runner/rerun_episodes.py, which skips finished
# episodes, so a preempted job can be requeued.
#
# The code, the tasks, and the outputs live on project storage (ROOT below), not
# in ~/scratch/prime_pilot: scratch is near its inode quota, and the rerun must
# use this corrected code, not the cluster's older copy.
#
# Usage (from ROOT):
#   sbatch -q <qos> -p gpu-h100 --gres=gpu:h100:1 --cpus-per-task=8 --mem=96G -t 3:00:00 \
#     --export=ALL,MODEL=Qwen/Qwen2.5-32B-Instruct,MODELTAG=qwen2.5-32b cluster/rerun_episodes.sbatch
# (TIERS defaults to "1 2 3"; pass it through the environment: env TIERS="1 2" sbatch --export=ALL ...)
set -euo pipefail
SCRATCH=$HOME/scratch
ROOT=<project>/prime_rerun
export HF_HOME=$SCRATCH/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=$SCRATCH/xdg_cache TRITON_CACHE_DIR=$SCRATCH/triton_cache
export VLLM_CACHE_ROOT=$SCRATCH/vllm_cache TORCHINDUCTOR_CACHE_DIR=$SCRATCH/inductor_cache
export FLASHINFER_WORKSPACE_BASE=$SCRATCH/flashinfer_ws TORCH_EXTENSIONS_DIR=$SCRATCH/torch_extensions
export CUDA_CACHE_PATH=$SCRATCH/nv_compute_cache VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_ENV=${VLLM_ENV:-$SCRATCH/envs/mllm}
export PATH="$VLLM_ENV/bin:$PATH"
MODEL=${MODEL:?set MODEL}
MODELTAG=${MODELTAG:?set MODELTAG}
TIERS=${TIERS:-1 2 3}
PARALLEL=${PARALLEL:-8}
case "${MODEL,,}" in
  *gpt-oss*|*gptoss*) PARSER=openai ;;
  *)                  PARSER=hermes ;;   # Qwen2.5
esac
PORT=$((8000 + SLURM_JOB_ID % 1000))
export PRIME_LLM_BASE=http://127.0.0.1:$PORT/v1 PRIME_LLM_KEY=local PRIME_LLM_MODEL=$MODEL
mkdir -p "$ROOT/logs"
echo "[job] model=$MODEL tag=$MODELTAG tiers=$TIERS parser=$PARSER node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-unset} git=$(cat $ROOT/CODE_COMMIT 2>/dev/null)"

$VLLM_ENV/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --tensor-parallel-size 1 --dtype auto --enforce-eager \
  --max-model-len 16384 --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser "$PARSER" \
  --port "$PORT" > "$ROOT/logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
HEALTHY=0
for i in $(seq 1 540); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo "[job] vllm healthy after $((i*10))s"; HEALTHY=1; break; fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then echo "[job] vllm died during startup"; tail -40 "$ROOT/logs/vllm-$SLURM_JOB_ID.log"; exit 1; fi
  sleep 10
done
[ "$HEALTHY" = 1 ] || { echo "[job] vllm never became healthy"; exit 1; }

cd "$ROOT"
_smoke() {
  curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"temperature\":0,\"tool_choice\":\"$1\",
         \"messages\":[{\"role\":\"user\",\"content\":\"$2\"}],
         \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"inspect_record\",
           \"description\":\"Inspect a field of the record\",
           \"parameters\":{\"type\":\"object\",\"properties\":{\"field_path\":{\"type\":\"string\",
             \"description\":\"dotted path of the field to read\"}},\"required\":[\"field_path\"]}}}]}" 2>/dev/null || echo '{}'
}
SMOKE=$(_smoke required "Read the field 'status' of the record.")
echo "$SMOKE" | grep -q '"tool_calls"' || { echo "[job] smoke (required) failed: $(echo "$SMOKE" | head -c 300)"; exit 2; }
SMOKE=$(_smoke auto "Call inspect_record with field_path set to 'status'. Do not ask any questions first.")
echo "$SMOKE" | grep -q '"tool_calls"' || { echo "[job] smoke (auto) failed: $(echo "$SMOKE" | head -c 300)"; exit 2; }
echo "[job] tool-calling smoke check: PASS"

python3 runner/rerun_episodes.py --modeltag "$MODELTAG" --tiers $TIERS --out rerun --parallel "$PARALLEL"
echo "[job] RERUN_JOB_DONE model=$MODEL tag=$MODELTAG"
