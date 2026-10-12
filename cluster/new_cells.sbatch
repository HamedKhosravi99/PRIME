#!/bin/bash
#SBATCH -J prime-cells
#SBATCH -A <account>
#SBATCH -N 1
#SBATCH --requeue
#SBATCH -o logs/%x-%j.out
# Run new cells (not reruns) with the corrected runner, served the way
# serve_and_run_guided.sbatch served the logged cells: vLLM with the parser
# chosen from the model id (Mistral: the mistral_safe plugin and vLLM's mistral
# tokenizer mode), dtype auto (bf16), TP 1, eager mode, 16k context, greedy
# decoding, and the same run_batch.py command line as its run_anchor helper.
# run_batch.py --resume skips finished episodes, so a preempted job can be
# requeued.
#
# CELLS lists "ARM:SPLIT:ANCHOR:OUTBASE" entries; each runs on the four domains
# into OUTBASE/<MODELTAG>-<domain>/<ARM>__<ANCHOR>/. Lists travel through the
# environment (sbatch splits --export values on commas):
#   env CELLS="react_program_attested:heldout_locator:bmin:results/w2loc" \
#     sbatch -q <qos> -p gpu-h100 --gres=gpu:h100:1 --cpus-per-task=8 --mem=96G -t 6:00:00 \
#     --export=ALL,MODEL=Qwen/Qwen2.5-32B-Instruct,MODELTAG=qwen2.5-32b cluster/new_cells.sbatch
set -euo pipefail
SCRATCH=$HOME/scratch
ROOT=<project>/prime_cells
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
CELLS=${CELLS:?set CELLS}
DOMAINS=${DOMAINS:-clinical security drugsafety financial}
PARALLEL=${PARALLEL:-8}
case "${MODEL,,}" in
  *mistral*|*mixtral*) PARSER=mistral_safe ;;
  *gpt-oss*|*gptoss*)  PARSER=openai ;;
  *)                   PARSER=hermes ;;   # Qwen2.5
esac
PLUGIN_FLAG=""; TOKMODE=""
if [ "$PARSER" = "mistral_safe" ]; then
  PLUGIN_FLAG="--tool-parser-plugin $ROOT/cluster/mistral_safe_parser.py"
  TOKMODE="--tokenizer-mode mistral"
fi
PORT=$((8000 + SLURM_JOB_ID % 1000))
export PRIME_LLM_BASE=http://127.0.0.1:$PORT/v1 PRIME_LLM_KEY=local PRIME_LLM_MODEL=$MODEL
mkdir -p "$ROOT/logs"
echo "[job] model=$MODEL tag=$MODELTAG parser=$PARSER cells=$CELLS node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-unset} git=$(cat $ROOT/CODE_COMMIT 2>/dev/null)"

$VLLM_ENV/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --tensor-parallel-size 1 --dtype auto --enforce-eager \
  --max-model-len 16384 --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser "$PARSER" $PLUGIN_FLAG $TOKMODE \
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

# two passes: the second reruns only episodes that ended in an error (for
# example a model-server timeout); finished episodes are read from their sidecars
for PASS in 1 2; do
for CELL in $CELLS; do
  IFS=: read -r ARM SPLIT ANCHOR OUTBASE <<< "$CELL"
  AA="--max-tool-calls 20"
  [ "$ANCHOR" != "B20" ] && AA="--max-tool-calls 20 --budget-anchor $ANCHOR"
  for DOM in $DOMAINS; do
    OUT="$OUTBASE/${MODELTAG}-${DOM}/${ARM}__${ANCHOR}"
    echo "[job] pass=$PASS arm=$ARM split=$SPLIT anchor=$ANCHOR out=$OUT start=$(date +%H:%M:%S)"
    python3 runner/run_batch.py --instances "bench/instances_v2/${DOM}/${SPLIT}/*.json" \
      --arm "$ARM" --license-source gold --model "$MODEL" --parallel "$PARALLEL" \
      $AA --resume --out "$OUT" || echo "[job] FAILED $ARM $ANCHOR $DOM (continuing)"
  done
done
done
echo "[job] CELLS_DONE model=$MODEL tag=$MODELTAG"
