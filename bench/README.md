# Benchmarks and tools

Everything here is stdlib Python or bash, and ran from `/root/gbench` on the container. The scripts talk to an OpenAI-compatible server on `127.0.0.1:8080` unless a port or URL is given.

## Servers

| Script | What it does |
|---|---|
| [`srv.sh`](srv.sh) | `srv.sh start NAME MODEL [serve args]` starts `ninferno-serve` on 8080 in the background, waits for `listening on`, and prints the plan and capacity lines. `srv.sh stop` stops it. The runners make copies with another binary path (`srv_final3.sh` and so on) |
| [`deploy_fast.sh`](deploy_fast.sh) | stops the 262K unit and any `srv.sh` server, starts `gemma4-31b-fast`, and waits until it answers |
| [`apply_candidate.sh`](apply_candidate.sh), [`revert_candidate.sh`](revert_candidate.sh) | the night-1 switch of both units to `apps-final3` with 7 drafts, with unit-file backups, and its rollback |
| [`deploy_final4.sh`](deploy_final4.sh) | the switch of both units to `apps-final4` (patch 08), with unit-file backups; restarts the fast unit at a moment with no request in flight |
| [`mtest_start.sh`](mtest_start.sh), [`mtest_end.sh`](mtest_end.sh) | the live test of patch 08: behind an idle gate, swap in the new build on 8080, send a UI chat, an API request and an API tool call, print the counters and the UI's view; then put the fast unit back |
| [`build_aside.sh`](build_aside.sh) | `build_aside.sh NAME` builds the tree and copies the binaries to `build/apps-NAME`, so experiments never replace the serving binary |
| [`idle_gate.sh`](idle_gate.sh), [`idle_gate_any.sh`](idle_gate_any.sh) | hold an overnight runner until the fast unit has served no request for N minutes and `/root/gbench/HOLD` does not exist |

## Measurements

| Script | What it measures | Output |
|---|---|---|
| [`gbench.py`](gbench.py) | prefill at given prompt sizes (unique-nonce prompts, server-reported tok/s), decode after 2K and 8K of context (greedy and sampled), and short chat (10 prompts × 512 tokens) | a `ROW` line appended to `results.jsonl` |
| [`geval.py`](geval.py) | GSM8K (200, chain of thought), MMLU (285), MMLU-Pro (280), HumanEval (164, executed); greedy | `quality.jsonl` plus per-item `qdetail_*.json` |
| [`needle.py`](needle.py) | three planted access codes in a Federal Register haystack, asks for one; prefill and decode speed at that length | `needle.jsonl`; [`needlesum.py`](needlesum.py) summarises a run log |
| [`verify.py`](verify.py) | `/v1/models`, a chat answer, streaming, and three agent-style turns on a long system prompt (cache reuse, latency) | stdout |
| [`turnprobe.py`](turnprobe.py) | time to first token of cached turns that append 60 to 4,000 characters | stdout |
| [`ppl.py`](ppl.py), [`scorecmp.py`](scorecmp.py) | perplexity from `ninferno-gemma4 score --dump` files; `scorecmp` compares two | stdout |
| [`lb.sh`](lb.sh) | NInferno's linear bench (built with `-DNINFERNO_BUILD_BENCHMARKS=ON`) on the six decode GEMM shapes at T 1, 9 and 16 | stdout |
| [`prof_nv.sh`](prof_nv.sh), [`kerntop.py`](kerntop.py) | an nsys kernel profile of a lab-tool prefill, grouped by kernel family | `prof_*_cuda_gpu_kern_sum.csv` and a summary |
| [`qmatrix.sh`](qmatrix.sh) | the quality suite on one artifact at 6 lanes (we used 4 for the 17+ GiB artifacts) | via `geval.py` |
| [`ggufmeta.py`](ggufmeta.py) | prints a GGUF's vocabulary and embedding metadata without numpy | stdout |

The `needle.py` haystack is NInferno's own `examples/cli/messages/long_niah_256k.json`. Its needles are seeded, so the same lengths and depths in the same order give the same prompts. That is how the 140K miss was reproduced on every build.

## Data

[`data/make_subsets.py`](data/make_subsets.py) rebuilds the four evaluation subsets from their public sources, with the MMLU-Pro ids from [`data/mmlupro280_ids.txt`](data/mmlupro280_ids.txt). It needs `pyarrow` for the MMLU-Pro parquet and checks that file's sha256. We checked the rebuilt files against the ones our runs used: identical items, answers and tests. The datasets themselves are not in this repository.

## Models

[`models/`](models/) has the download and conversion scripts we ran:
- the QAT-UD GGUF and the abliterated QAT BF16 (`dl2.sh`);
- the QAT assistant, the MTP GGUFs, DFlash, Pajari and Heretic (`dl3.sh`);
- Heretic through the QAT-aware encoder (`heretic.sh`);
- the QAT-NVFP4 checkpoint and its conversion (`nvfp4.sh`, `nvfp4_convert.sh`);
- the "MSE" NVFP4 repo (`nvfp4mse.sh`, which turned out identical);
- CUDA builds of llama.cpp and ik_llama.cpp for comparison (`build_engines.sh`).

## Runs

[`runs/`](runs/) keeps every queue that ran, as it ran.
- **Day 1, 2026-10-25** ([`runs/day1/`](runs/day1/)): `queue1.sh` to `queue29.sh`, the side builds (`build_*.sh`) and the kernel variant patchers (`patch_*.py`).
- **Night 1, 2026-10-25/26** ([`runs/night1/`](runs/night1/)): `night1.sh` to `night13.sh`, each behind the idle gate and ending with `deploy_fast.sh`, plus the trace analysis of agent turns (`turnsplit*.py`, `turnbins.py`).

Their logs are in [`results/runs/`](../results/runs/).
