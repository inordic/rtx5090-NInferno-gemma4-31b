# Results

All measured on the RTX 5090 Laptop GPU described in the [README](../README.md#what-runs), one request at a time unless stated. Rows are appended in the order they ran.

## Files

| File | Rows | What |
|---|---|---|
| [`results.jsonl`](results.jsonl) | one per [`gbench.py`](../bench/gbench.py) run | `ppN`: prefill tok/s for an N-token request (`ppN_ntok` is the exact prompt length). `tgNg` / `tgNs`: decode tok/s after N tokens of context, greedy / sampled. `chatg` / `chats`: short chat, greedy / sampled |
| [`quality.jsonl`](quality.jsonl) | one per [`geval.py`](../bench/geval.py) run | scores out of 200 / 280 / 285 / 164, plus seconds per suite |
| [`quality/`](quality/) | `qdetail_<label>.json` | per item: ok, gold, prediction (GSM8K also counts tokens). Pair two runs item by item to compare them |
| [`needle.jsonl`](needle.jsonl) | one per [`needle.py`](../bench/needle.py) run | per needle: length, depth, ok, prompt tokens, prefill and decode tok/s, answer |
| [`lb_base.txt`](lb_base.txt) | | NInferno's linear bench on the six decode GEMM shapes at T 1, 9 and 16, before patches 06 and 07 |
| [`profiles/`](profiles/) | nsys kernel summaries | `prof_nv_split`: NVFP4 prefill at 6.6K with the GeGLU split. `prof_i8_32k`: 32K prefill with INT8 attention. `prof_decode2`: a serve run with CUDA-graph nodes traced. `prof_q4_ws0/1`: Q4 prefill on ggml MMQ against the W4A8 tile |
| [`drafter/`](drafter/) | `spec-tune` logs and profiles | MTP acceptance and tokens per round for each drafter and depth, on math, code, prose and thinking |
| [`runs/`](runs/) | the logs of every queue and night runner in [`bench/runs/`](../bench/runs/) | |

## Labels you will see

| Label prefix | Build |
|---|---|
| `abl …` | the abliterated Q4_K the box served before: plain decode, 32K |
| `C2 … C8` | QAT-UD Q4 with Google's assistant (C2–C4) and the QAT assistant (C5–C8) |
| `P1 … P6` | QAT-UD Q4 prefill chunk sizes and the dense tuner |
| `L1`, `L2` | llama.cpp on the same QAT-UD GGUF, plain and with MTP 4 |
| `F1 … F9` | QAT-NVFP4: W4A4 (F1), the GeGLU split (F4, F5), INT8 attention (F9) |
| `P7`, `P8` | NVFP4 at chunk 1024 / 155K; W4A8 |
| `D64`, `D1` | GeGLU split at T ≥ 64 against every width (decode) |
| `Q0`, `Q1` | QAT-UD Q4 at 262K, split off / on |
| `UF` | the fast unit as deployed on day 1 (`apps-final`, 8 drafts) |
| `V0`, `V1` | W4A4 verify off / on |
| `K7 … K15`, `K6` | MTP draft counts |
| `DR7 …` | the shortlist-head drafters |
| `C2 candidate`, `C3 candidate` | `apps-final2` and `apps-final3` with 7 drafts |
| `Q4K7`, `Q4K8` | QAT-UD Q4 at 262K on `apps-final3`, 7 / 8 drafts |
| `LIVE fast unit after apply_candidate` | production after the night-1 switch |

## Reading the numbers

- **Decode varies with the text.** The drafter's acceptance depends on how predictable the continuation is, so `tg` values move ±5–10% between runs of the same build. `chat` (10 prompts) is the steadiest decode figure.
- **MMLU-Pro moves 147–152 on every NVFP4 build** from run to run, including W4A16. Treat differences inside that band as noise.
- **Perplexity values come from a chat model scored on raw text**, so they are high in absolute terms. Only differences between kernels on the same model count.
