# Software and hardware of the runs

## Serving the models

The agents talk to a model through an OpenAI-compatible endpoint served by vLLM. `experiments/slurm/serve_and_run_guided.sbatch` is the recipe the runs used. It sets:
- the tool-call parser from the model id;
- `--enable-auto-tool-choice`;
- greedy decoding;
- a 16,384-token context;
- eager mode.

| Model (Hugging Face id) | Where it was served | vLLM | Tool-call parser |
|---|---|---|---|
| `Qwen/Qwen2.5-32B-Instruct` | the confirmatory grid (Sections 6.1 and 6.2 with named fields): two A100 GPUs, tensor-parallel 2; the later runs (evidence location, textual evidence, the closest systems, the regulation study): one or two GPUs of the types the scheduler assigned (A100 80GB, H100, H200, RTX Pro Blackwell) | 0.24.0 (Python 3.11) | `hermes` |
| `Qwen/Qwen2.5-7B-Instruct` | one A100 GPU | 0.24.0 | `hermes` |
| `openai/gpt-oss-20b` | H200 GPUs | 0.24.0 | `openai` |
| `mistralai/Mistral-Small-24B-Instruct-2501` | security on H100 GPUs; clinical, drug safety, and financial on A100 80GB GPUs of a second cluster | 0.24.0; 0.10.1 on the second cluster | `mistral_safe` (`experiments/slurm/mistral_safe_parser.py`) |

The model weights were the repositories' default revisions at the time of the runs (August to October 2026). The revision hashes were not recorded.

For Mistral-Small-24B, the tokenizer mode is `mistral`, because vLLM 0.24 cannot use the tokenizer that transformers 5 returns. Its tool-call parser is the stock Mistral parser plus a repair step for one malformed output shape. Well-formed outputs take the stock path; see the file's docstring and Appendix C of the paper.

## The agent loop

`src/prime_agents/llm.py` calls the endpoint with the `openai` Python client (version 1.0 or later; the exact version on the cluster was not recorded). A failed request is retried at temperature 0. An episode whose requests still fail ends with an error and is left out of the rates; see `docs/KNOWN_ISSUES.md` and `experiments/analysis/episode_accounting.py`. Every run is greedy with one run per task, except the sampled-decoding check of Appendix E.1, which uses temperature 0.7 and three seeds.

## The checks of this release

The checks that need no model are:
- `make test`;
- `make corpus`;
- `make tables`;
- `make figures`;
- `make conformance`.

They were last run with Python 3.9.6 on macOS and these versions (formerly `requirements-lock.txt`):

```
numpy==1.26.4
matplotlib==3.9.4
pytest==8.4.2
```

The runtime, the scorer, and the generators use only the standard library. The continuous-integration workflow (`.github/workflows/ci.yml`) runs the same checks on Linux with Python 3.9 and 3.12.
