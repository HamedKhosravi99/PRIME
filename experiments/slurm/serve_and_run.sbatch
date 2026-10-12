#!/bin/bash
#SBATCH -J prime-pilot
#SBATCH -A <account>
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH -t 8:00:00
#SBATCH -o logs/%x-%j.out
# Usage: sbatch --export=ALL,MODEL=Qwen/Qwen2.5-7B-Instruct,TP=1 -p gpu-a100 --gres=gpu:a100:1 serve_and_run.sbatch
# The submitting wrapper (submit_all.sh) sets MODEL, TP, ARMS, PARALLEL.

set -euo pipefail
SCRATCH=$HOME/scratch
ROOT=$SCRATCH/prime_pilot
export HF_HOME=$SCRATCH/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # compute nodes have no internet; models are cached
[ -f "$ROOT/env.sh" ] && source "$ROOT/env.sh"   # bootstrap writes VLLM_ENV here
export VLLM_ENV=${VLLM_ENV:-$SCRATCH/envs/mllm}  # known-good: vllm 0.24.0, py3.11
export PRIME_LLM_BASE=http://127.0.0.1:8000/v1
export PRIME_LLM_KEY=local
export PRIME_LLM_MODEL=$MODEL
ARMS=${ARMS:-"direct rubric react prime"}
PARALLEL=${PARALLEL:-4}
LSRC=${LSRC:-gold}

mkdir -p $ROOT/logs $ROOT/results
echo "[job] model=$MODEL tp=$TP arms=$ARMS node=$(hostname) gpus=$CUDA_VISIBLE_DEVICES"

# 1. serve
$VLLM_ENV/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --tensor-parallel-size "$TP" \
  --max-model-len 16384 --gpu-memory-utilization 0.92 \
  --port 8000 > "$ROOT/logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null || true" EXIT

# 2. wait for health (up to 25 min for 72B load)
for i in $(seq 1 150); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then echo "[job] vllm healthy after ${i}0s"; break; fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then echo "[job] vllm died"; exit 1; fi
  sleep 10
done
curl -sf http://127.0.0.1:8000/health >/dev/null || { echo "[job] vllm never became healthy"; exit 1; }

# 3. run arms sequentially (episodes parallel within each arm)
cd $ROOT
for ARM in $ARMS; do
  echo "[job] === arm=$ARM license=$LSRC model=$MODEL ==="
  python3 src/prime_agents/run_batch.py \
    --instances "data/pilot/instances/swe/*.json" \
    --arm "$ARM" --license-source "$LSRC" \
    --model "$MODEL" --parallel "$PARALLEL" \
    --out "results/$(echo $MODEL | tr '/' '_')" || echo "[job] arm $ARM failed, continuing"
done
echo "[job] done"
