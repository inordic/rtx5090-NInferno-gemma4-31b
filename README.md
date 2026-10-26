# Gemma 4 31B on an RTX 5090 Laptop GPU with NInferno

This build serves Google's Gemma 4 31B (dense) from one 24 GB RTX 5090 Laptop GPU, in a Proxmox container, through [NInferno](https://github.com/Brontir/NInferno) with our patches. Its main client is an agent (Hermes). The headline numbers, measured server-side through the OpenAI API on port 8080 at one request at a time:

- **Decode: 118 tok/s on chat** (118.4 greedy / 114.0 sampled), 107–120 after 2K and 110–119 after 8K of context. That's 3.6× the 33 tok/s we started from. *Production since 2026-10-26* ([`results/results.jsonl`](results/results.jsonl), label `LIVE fast unit after apply_candidate`).
- **Prefill: 4,154 tok/s at 8K**, 3.4K at 32K, 2.6K at 64K and about 1.6K at 130K. We started from 1.4K.
- **Context: 155,000 tokens** in the fast profile and **262,144** (the model's full window) in the max-context profile, up from 32K.
- **Quality:** GSM8K 198/200, HumanEval 161/164, MMLU-Pro 147–152/280 (direct answer). The max-context profile (QAT Q4) scores MMLU-Pro 164–165/280; see [Quality](#quality).
- **Long context:** 9/9 needles at 32K, 67K and 130K on the fast profile, and 2/2 at 185K and 231K on the max-context profile.

Every number here comes from a file in [`results/`](results/), measured on the box described in [What runs](#what-runs). The night-by-night reports are in [`research/`](research/), and the build route, with a check after each step, is in [`BUILD.md`](BUILD.md).

What made the difference, in order of effect:
1. **A drafter matched to the target.** Google's QAT assistant drafts against the QAT target, converted losslessly with a QAT-aware Q4_0 encoder (our llama.cpp patch). That took decode from 33 to about 106 tok/s.
2. **K-only KV for the ten shared-KV full-attention layers.** Their V is unrope(K)/c, so only K is stored, in FP8. That's what fits 262K on 24 GB.
3. **NVFP4 W4A4 prefill, with the GeGLU projection split off a slow fused kernel.** Prefill went from 1.5K to 3.9K tok/s at 8K.
4. **FP8 attention for the 512-wide full-attention layers.** FP8 scores with an FP32-promoted accumulator and FP8 P·V give +12% prefill at 8K, +34% at 32K, +50% at 64K and +59% at 130K, at unchanged perplexity.
5. **Decode-step tuning.** 7 MTP drafts so the verify fits the 8-token kernel bucket, a better QKV schedule, the Q6 head on an 8-column tile, and the GeGLU split at every width: +8–18% decode.

## Speed

**Prefill.** One cold request with a unique prompt, server-reported prompt tok/s ([`results.jsonl`](results/results.jsonl)):

| Prompt | Fast profile (NVFP4, live) | Max-context profile (QAT Q4) | Start (abliterated Q4_K) |
|---:|---:|---:|---:|
| 2K | 3,067 | 1,427 | 1,335 |
| 8K | **4,154** | 1,531 | 1,409 |
| 32K | 3,438 | 1,332 | 1,272 (24K) |
| 64K | 2,579 | 1,092 | — |
| ~130K | ~1,600 | ~710 (before FP8 attention) | — |
| ~185K / ~231K | — | 613 / 528 | — |

**Decode.** One user, MTP with 7 drafts (greedy / sampled tok/s):

| Workload | Fast profile (live) | Max-context profile | Start |
|---|---:|---:|---:|
| Short chat, 10 prompts × 512 tokens | **118.4 / 114.0** | 106.8 / 99.7 | 33.2 |
| After 2K of context | 107.3 / 119.6 | 93.2 / 110.0 | 32.6 |
| After 8K of context | 110.3 / 118.9 | 106.2 / 104.7 | 32.3 |

Decode depends on how predictable the text is. The drafter lands about 7.4 tokens per round on code against about 2.5 on prose ([`results/drafter/`](results/drafter/)).

**Agent turns.** Each turn reuses the cached conversation. With a 17.6K-token system prompt, the first turn takes 5.2 s, and each later turn takes 0.5–0.6 s on the fast profile (0.65–0.73 s on the max-context profile). A turn's time to first token is about 0.3 s for a few dozen new tokens ([`bench/turnprobe.py`](bench/turnprobe.py), [night 1 report](research/2026-10-26-night1-decode.md#agent-turns-hermes-where-the-03-s-goes)).

**Tools.** OpenAI tool calls work, streamed and parallel included ([night 1 report](research/2026-10-26-night1-decode.md#tool-calling-works)). An agent loop with 20 tools ran 93 steps and 89 calls with none returned as text and no error, and replies after histories where the assistant wrote text with its calls came back as clean calls ([`bench/agenttest.py`](bench/agenttest.py), [`bench/textcalltest.py`](bench/textcalltest.py), [CHANGELOG](CHANGELOG.md)).

## Quality

Greedy decoding throughout, on the same 200 GSM8K, 280 MMLU-Pro, 285 MMLU (both direct letter answers) and 164 HumanEval items ([`bench/geval.py`](bench/geval.py); item lists in [`bench/data/`](bench/data/); per-item results in [`results/quality/`](results/quality/)):

| Build | GSM8K | MMLU-Pro | MMLU | HumanEval |
|---|---:|---:|---:|---:|
| Start: abliterated Q4_K | 197 | 161 | — | 159 |
| **Max-context profile: QAT-UD Q4** (FP8 KV, K-only KV or BF16 KV) | 196 | **164–165** | **248** | 159 |
| pajari Q4_0_V2 (same QAT weights) | 196 | 164 | — | 159 |
| **Fast profile: QAT-NVFP4**, day 1 (INT8 attention) | **198** | 150–152 | 244–245 | **161** |
| Fast profile, live (final3, 7 drafts) | **198** | 147 | — | **161** |
| QAT-NVFP4, W4A8 prefill | — | 148 | 246 | — |
| QAT-NVFP4, BF16 activations everywhere | — | 148 | 248 | — |

The NVFP4 build is faster and gives up about 13 MMLU-Pro questions against QAT Q4. The loss sits in the NVFP4 weights, not in our kernels:
- It is the same with 4-, 8- or 16-bit activations, and with BF16, INT8 or FP8 attention.
- NVFP4 (E2M1 codes with E4M3 block scales) can't hold the QAT model's int4 grid exactly.
- "MSE-scaled" NVFP4 doesn't help: melcheikh's MSE repo has byte-identical language-model tensors to the absmax one.
- On one 140K needle prompt, every NVFP4 build names the wrong one of three planted codes, and QAT Q4 names the right one; the other 24 NVFP4 needles at 32–140K pass.

Perplexity on a 32K window (lower is better) is how the attention kernels were judged. The scoring setup inflates absolute values; differences are what count:

| Full-attention kernel | Perplexity | Lab prefill 8K / 32K / 64K |
|---|---:|---:|
| BF16 scores, BF16 P·V (before) | 76.96 | 4,583 / 2,795 / 1,852 |
| INT8 scores, BF16 P·V | 78.56 | 4,706 / 3,021 / 2,071 |
| INT8 scores, FP8 P·V | 77.99 | 4,986 / 3,522 / 2,542 |
| FP8 scores, FP8 P·V, plain FP8 accumulator | 81.13 | 5,116 / 3,760 / 2,811 |
| **FP8 scores (promoted), FP8 P·V (deployed)** | **77.84** | **5,110 / 3,748 / 2,786** |

## What runs

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU (GB203, sm_120, 82 SMs, 24 GB GDDR7, 896 GB/s), driver 595.58.03, VBIOS 98.03.95.00.07 |
| Host | Proxmox VE, kernel 6.17.13-2-pve. The GPU is also passed to a second container; only one serves at a time |
| Container | Ubuntu 24.04, 8 cores, 48 GB RAM, CUDA 13.2 |
| Engine | NInferno ([Brontir/NInferno](https://github.com/Brontir/NInferno)) at `9895ebc` + [our patches 01–09](patches/ninferno/), CUDA arch `120a`, Release |
| Fast model | [`melcheikh/gemma-4-31B-it-qat-NVFP4-Blackwell`](https://huggingface.co/melcheikh/gemma-4-31B-it-qat-NVFP4-Blackwell), Google's QAT weights in NVFP4 (ModelOpt), converted with NInferno's `gemma4_31b_nvfp4` recipe (Q8 embedding table, Q6 head): 17.8 GiB |
| Max-context model | [`unsloth/gemma-4-31B-it-qat-GGUF`](https://huggingface.co/unsloth/gemma-4-31B-it-qat-GGUF) `UD-Q4_K_XL` (all Q4_0 inside, on the QAT grid): 16.1 GiB |
| Drafter | [`google/gemma-4-31B-it-qat-q4_0-unquantized-assistant`](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized-assistant), QAT-aware Q4_0 through [our llama.cpp patch](patches/llama.cpp/): 0.28 GB |
| Chat template | Google's Gemma 4 template of 2026-07-09 with an agent patch ([`config/templates/`](config/templates/)) |
| Speculative decoding | MTP, 7 draft tokens |
| KV cache | FP8. The full-attention layers store K only (`NINFERNO_GEMMA4_SHARED_VALUES=2`) |
| Prefill | W4A4 (fast profile), chunk 1024 (fast) or 512 (max context); GeGLU split at every width (`NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1`) |
| Serving | systemd units [`gemma4-31b-fast`](config/systemd/gemma4-31b-fast.service) (port 8080) and [`gemma4-31b-262k`](config/systemd/gemma4-31b-262k.service), which exclude each other |
| UI | the NInfer web UI 0.1.0 on ports 8081 and 8443; its Monitor shows every client of 8080 from the engine's `/metrics` ([`config/ninfer-ui/`](config/ninfer-ui/); the UI itself is not in this repository) |

## Repository layout

| Path | Contents |
|---|---|
| [`BUILD.md`](BUILD.md) | from a bare container to both profiles serving, step by step, with checks |
| [`CHANGELOG.md`](CHANGELOG.md) | every change to the running setup, newest first |
| [`patches/`](patches/) | our NInferno patches 01–09, the QAT-aware Q4_0 llama.cpp patch, and two env-gated experiments that did not pay |
| [`config/`](config/) | the systemd units of both profiles, the chat template they serve with, and the web UI's service files |
| [`bench/`](bench/) | the speed, quality, needle, agent-turn and kernel benchmarks, the model download and conversion scripts, the overnight runners with their idle gate, and every queue that ran |
| [`results/`](results/) | every measurement: speed and quality rows, per-item quality, needles, kernel profiles, drafter tuning, run logs |
| [`research/`](research/) | the engine and options research, the day-1 and night-1 reports, the QAT Q4 prefill design |

## Lessons from the way here

- **The drafter must match the target's quantization training.** Google's regular assistant accepted 0.37 of drafts on code against the QAT target; the QAT-paired assistant accepts 0.80–0.90. That alone took decode from 57 to 106 tok/s.
- **Convert QAT weights with a QAT-aware encoder.** A plain Q4_0 conversion re-derives each block's scale and moves weights off the QAT grid. Our encoder recovers the BF16 grid and stores it exactly, so 100% of the assistant's blocks round-trip.
- **Read the model's config before buying memory.** `attention_k_eq_v` plus a constant key-norm weight mean V can be rebuilt from K. Storing K only is what fits 262K.
- **A fused kernel can be the slow path.** NVFP4's fused GeGLU fell back to a generic W4A4 kernel. Splitting it took 8K prefill from 2.5K to 3.9K, and splitting it at decode widths too added 6–9% decode.
- **The FP8 tensor-core accumulator is not FP32.** FP8 scores raised perplexity by 4% until each MMA started from zero and was added in FP32. After that they beat INT8 on both speed and perplexity.
- **Kernel buckets matter.** A verify of 8 drafts runs 9 tokens and lands in the 16-token kernel bucket. 7 drafts stays in the 8-token bucket and is faster overall.
- **Find out where quality is lost before tuning kernels.** The NVFP4 knowledge gap stayed the same across every activation and attention variant; it is in the weights.
- **Agent turns are bounded by state snapshots, not tokenizing.** At 155K there is no room for device-side cached states, so each turn copies about 45 ms of state to the host. At 100K the plan keeps two on the device, and turns are 80–100 ms faster.

## Credits

This build stands on:
- **NInferno** by Brontir ([Brontir/NInferno](https://github.com/Brontir/NInferno), Apache-2.0), which started from **ninfer** by Neroued ([Neroued/ninfer](https://github.com/Neroued/ninfer)) and credits its other forks, among them iamwavecut's ninfer-all.
- **llama.cpp** by ggml-org ([ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), MIT), whose converter and Q4_0 encoder our patch extends.
- **Google**, for Gemma 4, the QAT checkpoints and the MTP assistants.
- **melcheikh**, for the QAT NVFP4 conversion (NVIDIA Model Optimizer); **Unsloth**, for the QAT UD GGUF; **Pajari**, for Q4_0_V2; **llmfan46**, for the Heretic QAT weights; **huihui-ai**, for the abliterated QAT weights we compared; and the authors of the abliterated Q4_K build this box served before.
- **iNordic**, for the NInfer web UI (Strata-derived).
- The benchmark authors: GSM8K (OpenAI), MMLU (Hendrycks et al.), MMLU-Pro (TIGER-Lab), HumanEval (OpenAI).

See [`NOTICE.md`](NOTICE.md) for licences.

## Licence

Our scripts, configs and documents are under MIT ([`LICENSE`](LICENSE)). The NInferno patches are under Apache-2.0, like NInferno, and the llama.cpp patch is under MIT, like llama.cpp. Third-party components keep their own licences; see [`NOTICE.md`](NOTICE.md). No model weights or datasets are included.

Not affiliated with Google, NVIDIA, the NInferno or ninfer authors, or ggml-org.
