#!/bin/bash
#SBATCH -J prime-guided
#SBATCH -A <account>
#SBATCH -q <qos>
#SBATCH -N 1
#SBATCH -t 8:00:00
#SBATCH --requeue
#SBATCH -o logs/%x-%j.out
#
# Obligation-guided-completion ablation (prime vs prime_guided vs
# static_contract) on an OPEN-WEIGHT model served locally by vLLM.
#
# Runs under the free QOS (usagefactor 0.0) because the <account>
# charge account is exhausted (0.03 available) -- <qos> is preemptible, so
# every run_batch call passes --resume and the job is --requeue-able: a
# preempted job resumes from its per-episode sidecars instead of restarting.
#
# Usage (from ~/scratch/prime_pilot/cluster/):
#   sbatch -p gpu-a100 --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G \
#     --export=ALL,MODEL=Qwen/Qwen2.5-7B-Instruct,TP=1,MODELTAG=qwen2.5-7b \
#     serve_and_run_guided.sbatch
#
# ALWAYS pass an explicit --mem. Do NOT use --mem=0 (the old serve_and_run.sbatch
# does): it requests the node's ENTIRE RAM, which no partially-allocated node can
# satisfy, so a <qos> job sits on "(Resources)" forever instead of backfilling.
#
# GPU requirement: this vLLM/PyTorch build supports compute capability >= 7.5.
# The gpu-v100 partition is CC 7.0 and CANNOT run it (verified: job 11564167 died
# in EngineCore init). Use gpu-a100 (CC 8.0) or gpu-h100 (CC 9.0) only.
#
# MODEL     HF id to serve (must be in $HF_HOME -- compute nodes are offline)
# TP        tensor-parallel size (= number of GPUs requested)
# MODELTAG  results dir prefix; analysis reads results/<MODELTAG>-<domain>/
# DOMAINS   "<domain-label>:<instance-subdir>" pairs (default: the four
#           judge-free domains the guided ablation targets)

set -euo pipefail
SCRATCH=$HOME/scratch
ROOT=$SCRATCH/prime_pilot
export HF_HOME=$SCRATCH/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # compute nodes have no internet
# Every compile cache goes to scratch. The home directory has a 20 GB quota
# and was full on 2026-10-09: Triton's kernel cache (default ~/.triton) hit
# "Disk quota exceeded" inside the engine's profile run, which killed the
# four Mistral servers at startup (13910697-703) and the four gpt-oss
# servers mid-run (13910689-95: every episode of every cell then errored
# and the job still printed GUIDED_RUN_DONE).
export XDG_CACHE_HOME=$SCRATCH/xdg_cache
export TRITON_CACHE_DIR=$SCRATCH/triton_cache
export VLLM_CACHE_ROOT=$SCRATCH/vllm_cache
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH/inductor_cache
# FlashInfer ignores XDG_CACHE_HOME and JIT-compiles into
# $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer (default: home). On H100/H200
# its kernels ship precompiled, so nothing was written; on the RTX PRO
# Blackwell nodes (sm_120) it compiled, hit the quota, and the three -h2
# twins died at engine start (13913562-64).
export FLASHINFER_WORKSPACE_BASE=$SCRATCH/flashinfer_ws
export TORCH_EXTENSIONS_DIR=$SCRATCH/torch_extensions
export CUDA_CACHE_PATH=$SCRATCH/nv_compute_cache
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" \
         "$FLASHINFER_WORKSPACE_BASE" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH"
export VLLM_ENV=${VLLM_ENV:-$SCRATCH/envs/mllm}  # known-good: vllm 0.24.0, py3.11
# The env's bin/ MUST be on PATH: vLLM's inductor backend shells out to `ninja`,
# which lives there and is NOT on the compute node's default PATH. Without this,
# EngineCore dies with FileNotFoundError: 'ninja' (verified: job 11564714).
export PATH="$VLLM_ENV/bin:$PATH"
# Software-review episodes shell out to `sys.executable -m pytest` inside the
# artifact repos (ToolExecutor.run_tests). Because the PATH export above puts
# the vLLM venv first, python3 (and sys.executable) IS the venv python -- and
# venv pythons ignore PYTHONUSERBASE/user-site entirely (verified: job
# 11566983's preflight refused the software domain for exactly this reason).
# pytest therefore lives IN the venv ($VLLM_ENV/bin/pip install pytest, done
# from the login node, which has internet). The preflight below still aborts
# the domain loudly if it is ever missing, so a silent all-abstention
# software artifact remains impossible.

MODEL=${MODEL:?set MODEL}
MODELTAG=${MODELTAG:?set MODELTAG}
TP=${TP:-1}
PARALLEL=${PARALLEL:-4}
BUDGET=${BUDGET:-20}
DOMAINS=${DOMAINS:-"research:res security:osv drugsafety:fda financial:fin"}
# Tool-call parsing is model-family specific, and getting it wrong does not
# error: vLLM returns prose, every arm emits nothing, and the run scores as
# a clean all-abstention result rather than an honest failure. Derived from
# the model id unless overridden.
case "${MODEL,,}" in
  *mistral*|*mixtral*) _P=mistral_safe ;;   # cluster/mistral_safe_parser.py, see PLUGIN_FLAG
  *llama-3*|*llama3*)  _P=llama3_json ;;
  *gpt-oss*|*gptoss*)  _P=openai ;;
  *olmo*)              _P=olmo3 ;;
  *phi-4*|*phi4*)      _P=hermes ;;   # phi-4 follows the Hermes convention
  *)                   _P=hermes ;;   # Qwen2.5 / Qwen3
esac
PARSER=${PARSER:-$_P}
# mistral_safe is the stock Mistral parser plus a repair step for the single-
# object [TOOL_CALLS] payload that Mistral-Small-24B emits now and then; the
# stock parser turns it into HTTP 400 and the episode dies on its first
# request (9/120 in job 13911412). Loaded as a vLLM tool-parser plugin.
PLUGIN_FLAG=""
[ "$PARSER" = "mistral_safe" ] && PLUGIN_FLAG="--tool-parser-plugin $ROOT/cluster/mistral_safe_parser.py"
# Mistral: transformers 5 returns a MistralCommonBackend from AutoTokenizer,
# which vLLM 0.24 cannot use ("CachedMistralCommonBackend has no attribute
# is_fast": jobs 11661349 and 13909467-73 died on it; "auto" is the default
# so the earlier --tokenizer-mode auto changed nothing, and the hf mode takes
# the same AutoTokenizer path). vLLM's own mistral tokenizer mode works
# (login-node check: MistralTokenizer, is_fast=True, chat template renders
# [AVAILABLE_TOOLS]); it reads tekken.json, which the cache has. Config and
# load format stay HF: the cache holds model-*.safetensors, not the
# consolidated files the mistral load format wants, and compute nodes are
# offline.
case "${MODEL,,}" in
  *mistral*|*mixtral*) TOKMODE="--tokenizer-mode mistral" ;;
  *)                   TOKMODE="" ;;
esac
DTYPE=${DTYPE:-auto}       # V100 (sm70) has no bfloat16: pass DTYPE=float16 there
# Eager by default: torch.compile/CUDA-graph capture is the single biggest
# startup-failure surface here (see the ninja note above) and this workload is
# latency-bound on a tool-call loop, not throughput-bound, so the compile win is
# not worth another 17-minute failed startup. Set EAGER=0 to re-enable compile.
EAGER=${EAGER:-1}
# Induction view for induced-license configs ("ids" = original identifier-only
# condition; "spec" = the specification view Definition 2 grants). Gold-license
# configs ignore it. See runner/run_batch.py --induction-view.
IVIEW=${INDUCTION_VIEW:-ids}
EAGER_FLAG=""; [ "$EAGER" = "1" ] && EAGER_FLAG="--enforce-eager"
# Distinct port per job so two rungs can share a node without clashing.
PORT=$((8000 + SLURM_JOB_ID % 1000))

export PRIME_LLM_BASE=http://127.0.0.1:$PORT/v1
export PRIME_LLM_KEY=local
export PRIME_LLM_MODEL=$MODEL

mkdir -p "$ROOT/logs" "$ROOT/results"
echo "[job] model=$MODEL tag=$MODELTAG tp=$TP port=$PORT node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[job] domains: $DOMAINS"

# ---- 1. serve -------------------------------------------------------------
# --enable-auto-tool-choice/--tool-call-parser are REQUIRED: runner/arms.py
# drives every arm through the OpenAI tools API, and vLLM otherwise ignores
# the `tools` field and returns prose (which would score as an all-abstention
# run rather than an honest failure).
$VLLM_ENV/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --tensor-parallel-size "$TP" --dtype "$DTYPE" $EAGER_FLAG \
  --max-model-len 16384 --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser "$PARSER" $PLUGIN_FLAG $TOKMODE \
  --port "$PORT" > "$ROOT/logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

# ---- 2. wait for health (up to 30 min for a large model) -------------------
HEALTHY=0
# 90 min. 30 was enough for one 32B server per node but not for three, and
# packed three shards onto one node, each reading a ~65GB model
# from shared scratch, and loading slowed to 152s per checkpoint shard. The
# third was at 12/17 when the timer expired -- it was progressing, not
# stuck. 60 min covers the contended case; a genuinely dead server still
# fails, just later, and the loop below already aborts early if the vllm
# process dies.
for i in $(seq 1 540); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[job] vllm healthy after $((i*10))s"; HEALTHY=1; break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "[job] vllm died during startup; tail of its log:"; tail -40 "$ROOT/logs/vllm-$SLURM_JOB_ID.log"; exit 1
  fi
  sleep 10
done
[ "$HEALTHY" = 1 ] || { echo "[job] vllm never became healthy"; tail -40 "$ROOT/logs/vllm-$SLURM_JOB_ID.log"; exit 1; }

# ---- 3. tool-calling smoke check ------------------------------------------
# Fail loudly HERE if the served model ignores the tools API, rather than
# silently producing a benchmark-wide abstention artifact.
#
# This tests the PLUMBING (does vLLM parse/emit tool calls at all), not the
# model's willingness to call a tool. Those are different things: a first
# version of this check asked "Look up record R1" without supplying the
# required argument, and Qwen2.5-7B correctly asked a clarifying question
# instead of calling the tool -- failing a perfectly healthy server. So:
# tier 1 forces the issue with tool_choice=required; tier 2 falls back to
# tool_choice=auto with a fully-specified instruction, in case this vLLM
# build rejects `required`. Only if BOTH produce no tool_calls is the tools
# API genuinely broken.
cd "$ROOT"
_smoke() {  # $1 = tool_choice value, $2 = user content
  curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"temperature\":0,\"tool_choice\":\"$1\",
         \"messages\":[{\"role\":\"user\",\"content\":\"$2\"}],
         \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"inspect_record\",
           \"description\":\"Inspect a field of the record\",
           \"parameters\":{\"type\":\"object\",\"properties\":{\"field_path\":{\"type\":\"string\",
             \"description\":\"dotted path of the field to read\"}},
           \"required\":[\"field_path\"]}}}]}" 2>/dev/null || echo '{}'
}
# Tier 1 proves the PLUMBING works. Tier 2 proves the MODEL WILL USE IT
# unprompted-by-force, and both must pass. microsoft/phi-4 passes tier 1 and
# fails tier 2: it has no native tool-calling behaviour, so it served a whole
# domain making ZERO tool calls in 120/120 episodes while every arm still
# "emitted" and the run looked superficially fine. A model that only calls a
# tool when the API forces it cannot do this benchmark, and a run under it is
# an artifact, not a result.
# Three attempts: job 13909461 (gpt-oss-20b) got an HTTP error on the very
# first request after the health check ('{}' is the curl-failure fallback)
# and aborted, while the identical server in 13909459 passed.
for _try in 1 2 3; do
  SMOKE=$(_smoke required "Read the field 'status' of the record.")
  echo "$SMOKE" | grep -q '"tool_calls"' && break
  echo "[job] smoke tier 1 attempt $_try produced no tool_calls: $(echo "$SMOKE" | head -c 200)"
  [ "$_try" -lt 3 ] && sleep 30
done
if ! echo "$SMOKE" | grep -q '"tool_calls"'; then
  echo "[job] smoke tier 1 (tool_choice=required) produced no tool_calls."
  echo "[job]   tier-1 response: $(echo "$SMOKE" | head -c 300)"
  echo "[job] aborting: the tools API is broken for this model/parser combination."
  exit 2
fi
SMOKE=$(_smoke auto "Call inspect_record with field_path set to 'status'. Do not ask any questions first.")
if ! echo "$SMOKE" | grep -q '"tool_calls"'; then
  echo "[job] tool-calling smoke check: FAIL -- plumbing works (tool_choice=required"
  echo "[job]   succeeded) but the model does NOT emit a tool call under"
  echo "[job]   tool_choice=auto even when told to. Every episode would gather no"
  echo "[job]   evidence, and the run would score as clean abstention rather than"
  echo "[job]   as the failure it is."
  echo "[job]   response was: $(echo "$SMOKE" | head -c 400)"
  echo "[job] aborting: MODEL=$MODEL PARSER=$PARSER is not usable for this benchmark."
  exit 2
fi
if echo "$SMOKE" | grep -q '"tool_calls"'; then
  echo "[job] tool-calling smoke check: PASS"
else
  echo "[job] tool-calling smoke check: FAIL -- server returned no tool_calls under either tier."
  echo "[job] response was: $(echo "$SMOKE" | head -c 600)"
  echo "[job] aborting: a run under a broken tools API would be an artifact, not a result."
  exit 2
fi

# ---- 4. the configured arm set, per domain --------------------------------
# CONFIGSET=guided   : the guided-completion ablation (tableC_guided_common.py
#                      GUIDED_CONFIGS) -- unguided/guided x gold/induced + static
# CONFIGSET=baseline : the review-requested fair-baseline comparison. Adds
#                      `masked_reactive` (same gold DNF routes, same graded
#                      lattice, same enforced mask, NO obligation frontier and
#                      NO minimal-route planning) plus `react` as the
#                      unconstrained reference, isolating PRIME's control law
#                      from its gate.
# CONFIGSET=all      : union of both, deduplicated.
CONFIGSET=${CONFIGSET:-guided}
case "$CONFIGSET" in
  guided)
    CONFIGS=("prime gold" "prime_guided gold" "prime induced" "prime_guided induced" "static_contract gold") ;;
  baseline)
    CONFIGS=("prime gold" "masked_reactive gold" "static_contract gold" "react gold" \
             "prime induced" "masked_reactive induced") ;;
  all)
    CONFIGS=("prime gold" "prime_guided gold" "masked_reactive gold" "static_contract gold" "react gold" \
             "prime induced" "prime_guided induced" "masked_reactive induced") ;;
  v2dev)
    # W0.6 LLM dev pilots on bench/instances_v2/*/dev (EXECUTION_PLAN v5):
    # G_dev (masked_reactive_v vs computed informed planner), TOST variance,
    # and the react_program instruction-effect pair. Budgets handled by the
    # driver loop below (BUDGETS env), gold license source only.
    CONFIGS=("masked_reactive_v gold" "prime_guided gold" "react_program gold" "react_program_verify gold" "react_program_attested gold" "react_verify gold" "react_spec gold" "react_spec_verify gold" "react gold") ;;
  w2alad)
    CONFIGS=("__ladder__ heldout") ;;
  w2apol)
    CONFIGS=("__pol__ heldout") ;;
  w2seed)
    # Decoding-variance replication. Own driver below.
    CONFIGS=("__seed__ gold") ;;
  w2ext)
    # External / scaffold baselines on the v2 benchmark. Own driver below.
    CONFIGS=("__ext__ gold") ;;
  w2extspec)
    # Information-matched reruns of the six comparators (review
    # response): same prompts plus the specification-view block. Own driver.
    CONFIGS=("__extspec__ gold") ;;
  w2text)
    CONFIGS=("__text__ gold") ;;
  w2flat)
    CONFIGS=("__flat__ gold") ;;
  w2extsys)
    # The closest external systems instantiated on the balanced split
    # (runner/extsys.py; EXT_ARMS and EXT_ANCHORS via the environment). Own
    # driver below.
    CONFIGS=("__extsys__ gold") ;;
  w2glean)
    # GLEAN as an offline verifier of the fully informed agent's logged
    # trajectories (runner/glean.py; needs results/w2a/*/react_program_attested
    # and results/v2dev/*/react_program_attested on this filesystem). Own
    # driver below.
    CONFIGS=("__glean__ gold") ;;
  w2fdaaa)
    CONFIGS=("__fdaaa__ gold") ;;
  w2fill)
    # Cells the multi-model table (app:tab:multimodel) marked with a dash:
    # the fully informed agent at b_min and 2b_min for every model, and for
    # Qwen the live locator completion at b_min and the locator mask at 20
    # calls. Own driver below.
    CONFIGS=("__fill__ gold") ;;
  w2multi)
    # A second and a third model on experiments A/B/C. Own driver below.
    # The first eight w2m jobs (13909459-73) died at the unknown-CONFIGSET
    # exit after a healthy vLLM and a passing smoke check: the driver block
    # existed but this case arm did not.
    CONFIGS=("__multi__ gold") ;;
  w2loc)
    # Locator-uncertain split (heldout_locator): PRIME and the mask at
    # b_min and 2b_min, PRIME at 20 calls. Own driver below.
    CONFIGS=("__loc__ gold") ;;
  w2complete)
    # PRIME with scripted completion (prime_complete) on heldout_balanced
    # at the four anchors. Own driver below.
    CONFIGS=("__complete__ gold") ;;
  w2ob)
    # One-branch-fails split (bench/instances_v2/*/heldout_onebranch): the
    # fully informed agent, the mask, and PRIME where exactly one of two
    # alternative routes to C2 survives. Own driver below.
    CONFIGS=("__ob__ gold") ;;
  s1v2)
    # W5: standard construction (S1) on the v2 benchmark. Own driver below.
    CONFIGS=("__s1__ spec") ;;
  w2bind)
    # W2b drives its own loop above; this arm exists only so the case does
    # not fall through to the unknown-CONFIGSET exit. Adding a driver block
    # without adding its case arm is how the first four W2b jobs died after
    # a healthy vLLM and a passing smoke check.
    CONFIGS=("__w2b__ induced") ;;
  w1ladder)
    # W1: the v5 ladder + re-costed guided on the LEGACY 61 instances
    # (bench/instances/{osv,fda,fin,ctgov,res}) at matched B=20 -- dev/
    # diagnostic material, not confirmatory W2a.
    CONFIGS=("react gold" "react_verify gold" "react_spec gold" "react_spec_verify gold" "react_program gold" "react_program_verify gold" "react_program_attested gold" "masked_reactive_v gold" "prime_guided gold") ;;
  compare)
    # review-response comparison set: the conformal-family instantiation
    # (confidence-elicited, analysis-time LOO risk control) plus the three
    # self-standard arms under 3 meaning-preserving scaffold variants each
    # (variant 0 = the published scaffold, byte-identical), turning
    # "adapter/prompt choices" into a measured sensitivity band.
    CONFIGS=("conformal gold") ;;
  *) echo "[job] unknown CONFIGSET=$CONFIGSET"; exit 3 ;;
esac
echo "[job] configset=$CONFIGSET (${#CONFIGS[@]} configs/domain)"

# ---- 4a0. W2a confirmatory shards (instances_v2 heldout splits) ------------
# ---- Decoding-variance replication ----------------------------------------
# The confirmatory runs are greedy and single-sample, so their intervals
# carry instance-clustering variance only. At temperature 0 a seed changes
# nothing, so the honest way to bound decoding sensitivity is to SAMPLE:
# three seeds at T=0.7 on the arms that carry the enforcement contrast.
if [ "$CONFIGSET" = "w2seed" ]; then
  for D in $DOMAINS; do
    DOM=${D%%:*}
    for ARM in react_program_attested masked_reactive_v prime_guided; do
      for SEED in 11 12 13; do
        OUT="results/w2seed/${MODELTAG}-${DOM}/${ARM}__T07S${SEED}__B20"
        python3 runner/run_batch.py --instances "bench/instances_v2/${DOM}/heldout_balanced/*.json" \
          --arm "$ARM" --license-source gold --model "$MODEL" --parallel "$PARALLEL" \
          --max-tool-calls 20 --temperature 0.7 --seed "$SEED" --resume --out "$OUT" \
          || echo "[job] FAILED $ARM seed=$SEED (continuing)"
      done
    done
  done
  echo "[job] GUIDED_RUN_DONE model=$MODEL tag=$MODELTAG configset=$CONFIGSET"
  exit 0
fi

# ---- External / scaffold baselines on the v2 benchmark --------------------
# W2a compared the ladder against its own rungs and the policy family, which
# are internal ablations. These are the independent comparators: an authored
# per-domain contract, two self-checking scaffolds, a calibrated-confidence
# gate, a prose rubric, and unconditioned direct answering.
if [ "$CONFIGSET" = "w2ext" ]; then
  for D in $DOMAINS; do
    DOM=${D%%:*}
    for ARM in static_contract cove reflexion conformal rubric direct; do
      OUT="results/w2ext/${MODELTAG}-${DOM}/${ARM}__B20"
      python3 runner/run_batch.py --instances "bench/instances_v2/${DOM}/heldout_balanced/*.json" \
        --arm "$ARM" --license-source gold --model "$MODEL" --parallel "$PARALLEL" \
        --max-tool-calls 20 --resume --out "$OUT" \
        || echo "[job] FAILED $ARM (continuing)"
    done
  done
  echo "[job] GUIDED_RUN_DONE model=$MODEL tag=$MODELTAG configset=$CONFIGSET"
  exit 0
fi

# ---- Locator-uncertain split and scripted completion ----------------------
run_anchor () {  # ARM SPLIT ANCHOR OUTBASE
  local ARM=$1 SPLIT=$2 ANCHOR=$3 OUTBASE=$4
  local AA="--max-tool-calls 20"
  [ "$ANCHOR" != "B20" ] && AA="--max-tool-calls 20 --budget-anchor $ANCHOR"
  local OUT="$OUTBASE/${MODELTAG}-${DOM}/${ARM}__${ANCHOR}"
  echo "[job] arm=$ARM split=$SPLIT anchor=$ANCHOR out=$OUT start=$(date +%H:%M:%S)"
  python3 runner/run_batch.py --instances "bench/instances_v2/${DOM}/${SPLIT}/*.json" \
    --arm "$ARM" --license-source gold --model "$MODEL" --parallel "$PARALLEL" \
    $AA --resume --out "$OUT" || echo "[job] FAILED $ARM $ANCHOR (continuing)"
}
if [ "$CONFIGSET" = "w2text" ]; then
  for D in $DOMAINS; do
    DOM=${D%%:*}
    for ARM in react_program_attested masked_reactive_v prime_guided; do
      run_anchor $ARM heldout_text B20 results/w2text
    done
  done
  echo "[job] GUIDED_RUN_DONE model=$MODEL tag=$MODELTAG configset=$CONFIGSET"; exit 0
fi
if [ "$CONFIGSET" = "w2flat" ]; then
  for D in $DOMAINS; do
    DOM=${D%%:*}
    run_anchor prime_flat heldout_balanced B20 results/w2a
  done
  echo "[job] GUIDED_RUN_DONE model=$MODEL tag=$MODELTAG configset=$CONFIGSET"; exit 0
fi
if [ "$CONFIGSET" = "w2extsys" ]; then
  # env EXT_ARMS="..." EXT_ANCHORS="B20 bmin" sbatch --export=ALL ... (sbatch
  # splits --export values on commas, so lists travel through the environment)
  for D in $DOMAINS; do
    DOM=${D%%:*}
    for ANCHOR in ${EXT_ANCHORS:-B20}; do
      for ARM in ${EXT_ARMS:-ext_hesp ext_prooforstop ext_researchloop ext_agentspec ext_agentspec_examine}; do
        run_anchor $ARM heldout_balanced $ANCHOR results/w2a
      done
    done
  done
  echo "[job] GUIDED_RUN_DONE model=$MODEL tag=$MODELTAG configset=$CONFIGSET"; exit 0
fi
if [ "$CONFIGSET" = "w2glean" ]; then
  for ANCHOR in ${EXT_ANCHORS:-B20}; do
    echo "[job] glean anchor=$ANCHOR start=$(date +%H:%M:%S)"
    python3 runner/glean.py --model "$MODEL" --modeltag "$MODELTAG" --anchor "$ANCHOR" \
      --calib-anchor B20 --parallel "${GLEAN_PARALLEL:-24}" || echo "[job] FAILED glean $ANCHOR (continuing)"
  done
  echo "[job] GUIDED_RUN_DONE model=$MODEL tag=$MODELTAG configset=$CONFIGSET"; exit 0
fi
if [ "$CONFIGSET" = "w2fill" ]; then
  for D in $DOMAINS; do
    DOM=${D%%:*}
    run_anchor react_program_attested heldout_balanced bmin results/w2a
    run_anchor react_program_attested heldout_balanced 2bmin results/w2a
    if [ "$MODELTAG" = "qwen2.5-32b" ]; then
      run_anchor prime_complete heldout_locator bmin results/w2loc
      run_anchor masked_reactive_v heldout_locator B20 results/w2loc
    fi
  done
  echo "[job] GUIDED_RUN_DONE model=$MODEL tag=$MODELTAG configset=$CONFIGSET"; exit 0
fi
if [ "$CONFIGSET" = "w2multi" ]; then
  # a second and a third model on the main-text experiments: the top rows of
  # the ladder (A), guidance and completion at three budgets (B), and the
  # locator-uncertain split (C), so the sign of each result is shown for
  # more than one model family
  for D in $DOMAINS; do
    DOM=${D%%:*}
    run_anchor react_program_attested heldout_balanced B20 results/w2a
    for ANCHOR in bmin 2bmin B20; do
      run_anchor masked_reactive_v heldout_balanced $ANCHOR results/w2a
      run_anchor prime_guided heldout_balanced $ANCHOR results/w2a
      run_anchor prime_complete heldout_balanced $ANCHOR results/w2a
    done
    for ANCHOR in bmin 2bmin; do
      run_anchor prime_guided heldout_locator $ANCHOR results/w2loc
    done
    run_anchor masked_reactive_v heldout_locator bmin results/w2loc
    run_anchor prime_complete heldout_locator bmin results/w2loc
  done
  echo "[job] GUIDED_RUN_DONE model=$MODEL tag=$MODELTAG configset=$CONFIGSET"; exit 0
fi
if [ "$CONFIGSET" = "w2fdaaa" ]; then
  for ARM in react_program_attested prime_guided prime_complete; do
    OUT="results/fdaaa/${MODELTAG}/${ARM}__B20"
    echo "[job] arm=$ARM split=fdaaa anchor=B20 out=$OUT start=$(date +%H:%M:%S)"
    python3 runner/run_batch.py --instances "bench/instances_fdaaa/fdaaa-*.json" \
      --arm "$ARM" --license-source gold --model "$MODEL" --parallel "$PARALLEL" \
      --max-tool-calls 20 --resume --out "$OUT" || echo "[job] FAILED $ARM (continuing)"
  done
  echo "[job] GUIDED_RUN_DONE model=$MODEL tag=$MODELTAG configset=$CONFIGSET"; exit 0
fi
if [ "$CONFIGSET" = "w2loc" ]; then
  for D in $DOMAINS; do
    DOM=${D%%:*}
    for ANCHOR in bmin 2bmin; do
      run_anchor prime_guided heldout_locator $ANCHOR results/w2loc
      run_anchor masked_reactive_v heldout_locator $ANCHOR results/w2loc
    done
    run_anchor prime_guided heldout_locator B20 results/w2loc
  done
  echo "[job] GUIDED_RUN_DONE model=$MODEL tag=$MODELTAG configset=$CONFIGSET"
  exit 0
fi
if [ "$CONFIGSET" = "w2complete" ]; then
  for D in $DOMAINS; do
    DOM=${D%%:*}
    for ANCHOR in bmin bmin1 2bmin B20; do
      run_anchor prime_complete heldout_balanced $ANCHOR results/w2a
    done
  done
  echo "[job] GUIDED_RUN_DONE model=$MODEL tag=$MODELTAG configset=$CONFIGSET"
  exit 0
fi

# ---- Information-matched comparators (review response) --------------------
# Identical to w2ext except that every comparator also receives the
# specification view (PRIME_COMPARATOR_SPEC=1); outputs carry a __spec tag so
# the published directories are never touched. Table: analysis/build_ext_table.py --variant spec
if [ "$CONFIGSET" = "w2extspec" ]; then
  for D in $DOMAINS; do
    DOM=${D%%:*}
    for ARM in static_contract cove reflexion conformal rubric direct; do
      OUT="results/w2ext/${MODELTAG}-${DOM}/${ARM}__spec__B20"
      python3 runner/run_batch.py --instances "bench/instances_v2/${DOM}/heldout_balanced/*.json" \
        --arm "$ARM" --license-source gold --model "$MODEL" --parallel "$PARALLEL" \
        --max-tool-calls 20 --comparator-spec-view --resume --out "$OUT" \
        || echo "[job] FAILED $ARM (continuing)"
    done
  done
  echo "[job] GUIDED_RUN_DONE model=$MODEL tag=$MODELTAG configset=$CONFIGSET"
  exit 0
fi

# ---- One-branch-fails split (review response) ------------------------------
# Generated locally by bench/generate_record_instances.py --one-branch-per-domain
# and synced with the rest of bench/instances_v2. Table: analysis/build_onebranch_table.py
if [ "$CONFIGSET" = "w2ob" ]; then
  for D in $DOMAINS; do
    DOM=${D%%:*}
    for ARM in react_program_attested masked_reactive_v prime_guided; do
      OUT="results/w2ob/${MODELTAG}-${DOM}/${ARM}__B20"
      python3 runner/run_batch.py --instances "bench/instances_v2/${DOM}/heldout_onebranch/*.json" \
        --arm "$ARM" --license-source gold --model "$MODEL" --parallel "$PARALLEL" \
        --max-tool-calls 20 --resume --out "$OUT" \
        || echo "[job] FAILED $ARM (continuing)"
    done
  done
  echo "[job] GUIDED_RUN_DONE model=$MODEL tag=$MODELTAG configset=$CONFIGSET"
  exit 0
fi

# ---- W5: standard construction (S1) on the v2 benchmark -------------------
# The legacy OPENVOCAB block (section 6) is pinned to the old instance
# layout AND is unreachable from any v2 configset, since the driver blocks
# exit before it. This runs S1 against instances_v2 so the S1 claim is about
# the benchmark this paper reports rather than the earlier instance family.
if [ "$CONFIGSET" = "s1v2" ]; then
  S1SPLIT=${S1_SPLIT:-heldout_balanced}
  S1LIMIT=${S1_LIMIT:-30}
  for D in $DOMAINS; do
    DOM=${D%%:*}
    python3 analysis/openvocab_endtoend.py \
      --domains "${DOM}:${DOM}" --v2-split "$S1SPLIT" --limit "$S1LIMIT" \
      --model "$MODEL" --tag "${MODELTAG}-v2" --budget 20 \
      || echo "[job] FAILED S1 domain=$DOM (continuing)"
  done
  echo "[job] GUIDED_RUN_DONE model=$MODEL tag=$MODELTAG configset=$CONFIGSET"
  exit 0
fi

# ---- W2b induced-program runs (W3 conservative induction) -----------------
# The induced condition is the ONLY place the W3 stack is exercised
# end-to-end. --induction-method state_first is mandatory here: the default
# is the published one-shot condition, and running W2b without this flag
# would measure the legacy induction under the new label.
if [ "$CONFIGSET" = "w2bind" ]; then
  for D in $DOMAINS; do
    DOM=${D%%:*}
    for ARM in masked_reactive_v prime_guided_gated prime_guided; do
      OUT="results/w2b/${MODELTAG}-${DOM}/${ARM}__induced__B20"
      python3 runner/run_batch.py --instances "bench/instances_v2/${DOM}/heldout_balanced/*.json" \
        --arm "$ARM" --license-source induced --induction-method state_first \
        --model "$MODEL" --parallel "$PARALLEL" --max-tool-calls 20 --resume \
        --out "$OUT" || echo "[job] FAILED $ARM (continuing)"
    done
  done
  echo "[job] GUIDED_RUN_DONE model=$MODEL tag=$MODELTAG configset=$CONFIGSET"
  exit 0
fi

if [ "$CONFIGSET" = "w2alad" ] || [ "$CONFIGSET" = "w2apol" ]; then
  for D in $DOMAINS; do
    DOM=${D%%:*}
    if [ "$CONFIGSET" = "w2alad" ]; then
      for ARM in react react_verify react_spec react_spec_verify react_program react_program_verify react_program_attested masked_reactive_v prime_guided; do
        OUT="results/w2a/${MODELTAG}-${DOM}/${ARM}__B20"
        python3 runner/run_batch.py --instances "bench/instances_v2/${DOM}/heldout_balanced/*.json"           --arm "$ARM" --license-source gold --model "$MODEL" --parallel "$PARALLEL"           --max-tool-calls 20 --resume --out "$OUT" || echo "[job] FAILED $ARM (continuing)"
      done
      for ARM in react react_program_attested masked_reactive_v prime_guided; do
        OUT="results/w2a/${MODELTAG}-${DOM}/natural__${ARM}__B20"
        python3 runner/run_batch.py --instances "bench/instances_v2/${DOM}/heldout_natural/*.json"           --arm "$ARM" --license-source gold --model "$MODEL" --parallel "$PARALLEL"           --max-tool-calls 20 --resume --out "$OUT" || echo "[job] FAILED natural $ARM (continuing)"
      done
    else
      for ARM in masked_reactive_v prime_guided; do
        for ANCHOR in bmin bmin1 2bmin; do
          OUT="results/w2a/${MODELTAG}-${DOM}/${ARM}__${ANCHOR}"
          python3 runner/run_batch.py --instances "bench/instances_v2/${DOM}/heldout_balanced/*.json"             --arm "$ARM" --license-source gold --model "$MODEL" --parallel "$PARALLEL"             --max-tool-calls 20 --budget-anchor "$ANCHOR" --resume --out "$OUT" || echo "[job] FAILED $ARM $ANCHOR (continuing)"
        done
      done
    fi
  done
  echo "[job] GUIDED_RUN_DONE model=$MODEL tag=$MODELTAG configset=$CONFIGSET"
  exit 0
fi

for D in $DOMAINS; do
  # configsets with their own driver loops must not enter the legacy loop:
  # its NINST=$(ls bench/instances/... | wc -l) fails under set -e/pipefail
  # for v2 domains (paths live under bench/instances_v2), which killed the
  # first W2a shards at launch.
  case "$CONFIGSET" in v2dev|w2alad|w2apol|w2bind|s1v2|w2ext|w2seed) break ;; esac
  DOM=${D%%:*}; SUB=${D##*:}
  NINST=$(ls bench/instances/$SUB/*.json 2>/dev/null | wc -l || true)
  echo "[job] ===== domain=$DOM (bench/instances/$SUB, $NINST instances) ====="
  if [ "$SUB" = "swe" ] && ! python3 -c "import pytest" 2>/dev/null; then
    echo "[job] PREFLIGHT FAIL: pytest not importable -- skipping software domain"
    echo "[job] (a run without pytest would be an all-abstention artifact, not a result)"
    continue
  fi
  for CFG in "${CONFIGS[@]}"; do
    ARM=${CFG%% *}; LSRC=${CFG##* }
    OUT="results/${MODELTAG}-${DOM}/${ARM}__${LSRC}"
    # w1ladder: v5 call-class semantics -> NEVER resume into old dirs
    [ "$CONFIGSET" = "w1ladder" ] && OUT="results/w1v5/${MODELTAG}-${DOM}/${ARM}__${LSRC}"
    echo "[job] --- arm=$ARM license=$LSRC -> $OUT ---"
    python3 runner/run_batch.py \
      --instances "bench/instances/$SUB/*.json" \
      --arm "$ARM" --license-source "$LSRC" --induction-view "$IVIEW" \
      --model "$MODEL" --parallel "$PARALLEL" \
      --max-tool-calls "$BUDGET" --resume --out "$OUT" \
      || echo "[job] FAILED arm=$ARM lsrc=$LSRC domain=$DOM (continuing)"
  done
done

# ---- 4a. v2dev configset: budget-swept dev runs on instances_v2 -----------
if [ "$CONFIGSET" = "v2dev" ]; then
  BUDGETS=${BUDGETS:-"3 4 20"}
  for D in $DOMAINS; do
    DOM=${D%%:*}; SUB=${D##*:}
    for CFG in "${CONFIGS[@]}"; do
      ARM=${CFG%% *}; LSRC=${CFG##* }
      ARMBUDGETS="$BUDGETS"
      case "$ARM" in
        masked_reactive_v|prime_guided) ;;   # full sweep (G_dev / TOST)
        *) ARMBUDGETS="20" ;;                # ladder arms: matched budget only
      esac
      for B in $ARMBUDGETS; do
        OUT="results/v2dev/${MODELTAG}-${DOM}/${ARM}__B${B}"
        echo "[job] --- v2dev arm=$ARM B=$B domain=$DOM -> $OUT ---"
        python3 runner/run_batch.py \
          --instances "bench/instances_v2/${SUB}/dev/*.json" \
          --arm "$ARM" --license-source "$LSRC" \
          --model "$MODEL" --parallel "$PARALLEL" \
          --max-tool-calls "$B" --resume --out "$OUT" \
          || echo "[job] FAILED v2dev arm=$ARM B=$B domain=$DOM (continuing)"
      done
    done
  done
fi

# ---- 4b. compare configset only: scaffold-variant sensitivity runs --------
if [ "$CONFIGSET" = "compare" ]; then
  for D in $DOMAINS; do
    DOM=${D%%:*}; SUB=${D##*:}
    for ARM in rubric reflexion cove; do
      VMAX=2; [ "$ARM" = "reflexion" ] && VMAX=3   # v3 = verbatim released Reflexion prompt
      for V in $(seq 0 $VMAX); do
        OUT="results/${MODELTAG}-${DOM}/${ARM}_v${V}__gold"
        echo "[job] --- scaffold arm=$ARM variant=$V -> $OUT ---"
        PRIME_SCAFFOLD_VARIANT=$V python3 runner/run_batch.py \
          --instances "bench/instances/$SUB/*.json" \
          --arm "$ARM" --license-source gold \
          --model "$MODEL" --parallel "$PARALLEL" \
          --max-tool-calls "$BUDGET" --resume --out "$OUT" \
          || echo "[job] FAILED scaffold arm=$ARM v=$V domain=$DOM (continuing)"
      done
    done
  done
fi

# ---- 5. optional: vocabulary-condition experiment (VOCAB=1) ---------------
# Measures how much of the evaluation standard is actually learned, under
# three nested conditions (ids / spec / open vocabulary). Judge-free.
if [ "${VOCAB:-0}" = "1" ]; then
  VCONDS=${VOCAB_CONDITIONS:-"ids spec open"}
  VTAG=${VOCAB_TAG:-$MODELTAG}
  VOUT=${VOCAB_OUT_PREFIX:-vocabulary_conditions}
  for D in $DOMAINS; do
    DOM=${D%%:*}; SUB=${D##*:}
    echo "[job] ===== vocabulary conditions ($VCONDS), domain=$DOM ====="
    # T4's vocabulary panel must come from the SAME benchmark the rest of
    # the v2 results use. This was hardcoded to the legacy layout, so a
    # VOCAB run would have populated the panel from the old instance set
    # while every neighbouring number came from instances_v2.
    if [ -n "${VOCAB_V2_SPLIT:-}" ]; then
      VGLOB="bench/instances_v2/$DOM/${VOCAB_V2_SPLIT}/*.json"
    else
      VGLOB="bench/instances/$SUB/*.json"
    fi
    python3 analysis/vocabulary_conditions.py \
      --instances "$VGLOB" \
      --conditions $VCONDS \
      --model "$MODEL" --tag "${VTAG}-${DOM}" \
      --out "results/${VOUT}_${VTAG}_${DOM}.json" \
      || echo "[job] FAILED vocabulary conditions domain=$DOM (continuing)"
  done
fi

# ---- 6. optional: standard-construction (S1) end-to-end (OPENVOCAB=1) -----
# Open-vocabulary standard construction run through the UNCHANGED masked
# runtime, scored against the authored standard by evidence signatures only
# (analysis/openvocab_endtoend.py). Record-family domains only.
if [ "${OPENVOCAB:-0}" = "1" ]; then
  OVTAG=${OPENVOCAB_TAG:-$MODELTAG}
  python3 analysis/openvocab_endtoend.py \
    --domains security:osv drugsafety:fda financial:fin clinical:ctgov \
    --model "$MODEL" --tag "$OVTAG" --budget "$BUDGET" \
    || echo "[job] FAILED openvocab end-to-end (continuing)"
fi

echo "[job] GUIDED_RUN_DONE model=$MODEL tag=$MODELTAG"
