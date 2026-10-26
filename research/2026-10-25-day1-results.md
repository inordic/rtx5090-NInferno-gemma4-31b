# Day 1 (2026-10-25): the build

*A snapshot at the end of day 1. Night 1 then added 8–18% decode; see [the night-1 report](2026-10-26-night1-decode.md) and the [README](../README.md) for the current numbers.*

## Bottom line

Gemma 4 31B (dense) now decodes at **~105–130 tok/s** and prefills at **4.2K tok/s at 8K tokens** on the single 24 GB RTX 5090 Laptop GPU. It holds **155K tokens of context** in the fast profile and **262K** in the max-context profile. The day started with the abliterated Q4_K model on port 8080 at 1.4K tok/s prefill, 33 tok/s decode and 32K of context: **3x prefill, 3.3x decode, 5–8x context.**

The fast profile now serves port 8080 as `gemma4-31b-fast.service`. The max-context profile is installed next to it as `gemma4-31b-262k.service`, and the two units exclude each other. All numbers below are server-side through the OpenAI API on 8080 at one request at a time. Decode is greedy / sampled.

| | Start (abl Q4_K) | Fast (QAT-NVFP4) | Max context (QAT Q4) |
|---|---|---|---|
| Prefill 2K / 8K (tok/s) | 1,335 / 1,409 | **3,047 / 4,248** | 1,434 / 1,558 |
| Prefill 32K / 64K (tok/s) | 1,272 (24K) / — | **3,449 / 2,591** | 1,332 / 1,092 |
| Prefill at ~130K (tok/s) | — | ~1,600 | ~700 |
| Decode, chat (tok/s) | 33.2 | **108.6 / 105.6** | 105.0 / 102.9 |
| Decode after 2K / 8K context (tok/s) | 32.6 / 32.3 | **117 / 129, 98 / 104** | 101 / 115, 99 / 85 |
| First turn with a 17.6K-token system prompt | — | 5.2 s | 13.1 s |
| Next turns (17.6K tokens reused from cache) | — | 0.5–0.6 s | 0.65–0.73 s |
| Context | 32K | 155K | 262K |
| GSM8K | 197/200 | 198/200 | 196/200 |
| HumanEval | 159/164 | 161/164 | 159/164 |
| MMLU-Pro (direct answer) | 161/280 | 150–152/280 | 164–165/280 |
| MMLU (direct answer) | — | 244–248/285 | 248/285 |

The two profiles trade knowledge accuracy for speed. Every NVFP4 variant scores 148–152/280 on direct-answer MMLU-Pro against 164–165 for the QAT Q4 build, and NVFP4 missed one 140K needle that Q4 found. GSM8K, HumanEval and plain MMLU are equal or slightly better on NVFP4.

## What moved the numbers

Five changes, all in NInferno, plus one llama.cpp patch for the drafter conversion:

| Change | Effect |
|---|---|
| QAT-matched MTP drafter (`gemma-4-31B-it-qat-q4_0-unquantized-assistant`), losslessly converted, 8 draft tokens | Decode 33 → ~100 tok/s |
| K-only KV for the ten k_eq_v full-attention layers (V = unrope(K) / c) | Context 32K → 262K (Q4) / 155K (NVFP4) |
| NVFP4 W4A4 weights, with the GeGLU projection split off the slow fused kernel | Prefill 1.5K → 3.9K tok/s at 8K |
| FP8 Q·K and FP8 P·V in the 512-wide full-attention kernel | Prefill +12% at 8K, +34% at 32K, +50% at 64K, +59% at ~130K |
| GeGLU split at every batch width, so the verify pass uses the NVFP4 GEMM kernel | Decode +6–9% |

### Decode: the drafter had to match the QAT target

Google's regular assistant accepted only 0.37 of drafts on code against the QAT target at depth 3. The QAT-paired assistant accepts 0.90 at depth 4. A naive llama.cpp Q4_0 conversion breaks the QAT grid, so a QAT-aware Q4_0 encoder (patch 0001, llama.cpp) recovers each block's BF16 scale and keeps the drafter lossless. With 8 draft tokens the lab tool measured, on the QAT-aware Q4_0 Heretic target:

| Task | tok/s | Tokens per round | Acceptance |
|---|---|---|---|
| Code | 171.5 | 7.43 | 0.80 |
| Math | 152.8 | 6.63 | 0.70 |
| Thinking | 152.7 | 6.57 | 0.70 |
| Prose | 58.3 | 2.49 | 0.19 |

Served end to end, short chat decodes at ~100–108 tok/s.

The decode profile then showed where a verify step goes, at ~43 ms per step:

- **The gate/up projection: 40%.** It ran through a generic fused GeGLU kernel at ~440 GB/s, while the dedicated NVFP4 A16 kernel pulls ~545 GB/s on the other projections.
- **The rest:** the down projection 16%, the Q6 output head 6%, the drafter 9%, and attention with its split-K combine 5%.

Routing every width through the split (`NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1`) gave **+6–9% decode** (chat 101.9 → 108.6 greedy, 97.8 → 106.2 sampled).

### Context: store K once for the full layers

Gemma 4 31B's ten full-attention layers share K and V (`attention_k_eq_v`). Their key_norm weight is one constant c ≈ 0.0615, so V = unrope(K) / c. The engine now stores only K in FP8 and rebuilds V inside the attention kernel. That halves the full layers' KV:

- **QAT Q4:** the full 262K context fits with the drafter at 8 tokens and chunk 512.
- **NVFP4:** its weights are 1.7 GiB larger, so the ceiling depends on the prefill chunk: 92.7K at chunk 2048, 155.8K at chunk 1024, 185.3K at chunk 512.

Chunk 1024 costs no prefill speed (P7 vs F9: 3,925 vs 4,127 tok/s at 8K, within noise, and equal at 32K and 64K), so the fast profile runs 155K.

### Prefill: NVFP4 W4A4, then attention

The QAT Q4 weights prefill through INT8 MMQ at ~1.5K tok/s. melcheikh's NVFP4 conversion of the same QAT checkpoint runs native FP4 block-scaled MMA (W4A4). It reached only 2.5K at first. The fused GeGLU on prefill fell back to a generic W4A4 kernel; splitting it into the NVFP4 GEMM plus a separate GELU·mul took 8K prefill to **3.9K tok/s**.

Past 16K tokens the ten full-attention layers dominate. At 32K they took 52% of GPU time against 33% for all the GEMMs. Three successive kernel changes, perplexity on a 32K window (lower is better):

| Attention variant | Perplexity | 8K / 32K / 64K prefill (lab, tok/s) |
|---|---|---|
| BF16 scores, BF16 P·V (before) | 76.96 | 4,583 / 2,795 / 1,852 |
| INT8 scores, BF16 P·V | 78.56 | 4,706 / 3,021 / 2,071 |
| INT8 scores, FP8 P·V | 77.99 | 4,986 / 3,522 / 2,542 |
| FP8 scores, FP8 P·V, one accumulator | 81.13 | 5,116 / 3,760 / 2,811 |
| **FP8 scores (promoted), FP8 P·V (deployed)** | **77.84** | **5,110 / 3,748 / 2,786** |

The FP8 tensor-core accumulator keeps fewer bits than FP32. Starting each Q·K MMA from zero and adding into FP32 registers fixed the 81.1 regression. The K codes feed the MMA exactly as stored. On the 129K needle prompt, prefill went from 1,010 tok/s (BF16) to 1,609 (FP8).

## Quality

All runs are greedy, using the same 200 GSM8K, 280 MMLU-Pro, 285 MMLU (direct letter answers) and 164 HumanEval items.

| Build | GSM8K | MMLU-Pro | MMLU | HumanEval |
|---|---|---|---|---|
| abl Q4_K (the old 8080 model) | 197 | 161 | — | 159 |
| QAT-UD Q4_K_XL (fp8 KV / K-only / bf16 KV) | 196 | 164–165 | 248 | 159 |
| pajari Q4_0_V2 | 196 | 164 | — | 159 |
| QAT-NVFP4, W4A4 prefill | 198 | 150–152 | 244–245 | 161 |
| QAT-NVFP4, W4A8 prefill | — | 148 | 246 | — |
| QAT-NVFP4, BF16 activations everywhere | — | 148 | 248 | — |

The NVFP4 MMLU-Pro gap comes from the weights, not the speed tricks. It is the same with 8-bit and 16-bit activations, with or without INT8/FP8 attention.

melcheikh's "MSE-scaled" NVFP4 repo turned out to hold byte-identical language-model tensors to the absmax repo, so it changes nothing.

The 140K needle miss reproduces the same way: on the exact prompt, every NVFP4 build (BF16, INT8 or FP8 attention) names the wrong one of three planted codes, and QAT Q4 names the right one. On 24 other needles at 32–140K, NVFP4 scored 24/24.

## Long context

Needles are three planted access codes per document, asking for one:

| Build | ~30K | ~67K | 129–131K |
|---|---|---|---|
| **Deployed fast unit (NVFP4 + FP8 attention)** | **3/3** | **3/3** | **3/3** |
| NVFP4 + INT8 attention | 3/3 | 3/3 | 3/3, plus the 1 miss |
| NVFP4, BF16 attention | — | — | 3/3, plus the same miss |
| NVFP4 + FP8 attention (earlier build) | — | — | 3/3, plus the same miss |
| QAT Q4 | — | — | 4/4 (incl. the miss prompt) |
| QAT Q4 at 162–231K | — | — | 5/5 |

On the deployed unit, prefill ran ~3,400 tok/s at 30K, 2,380 at 67K and 1,565 at 130K.

## What did not work

- **W4A8 NVFP4 prefill (new `--activations int8` for Gemma 4):** 900 tok/s, because the engine's W4A8 NVFP4 route is slow. MMLU-Pro stayed at 148, so activations are not where NVFP4 loses accuracy.
- **INT8 scores on the 256-wide sliding layers:** perplexity got worse. INT8 and later FP8 stay on the 512-wide full layers only.
- **Split-Q FP8 scores (two E4M3 parts):** worse, 82.6 and 86.5. Promotion alone fixed the precision.
- **Other engines:** llama.cpp (QAT-UD, MTP 4) reached 97 tok/s decode and 1.3K prefill but tops out near 128K context. vLLM, SGLang and TensorRT-LLM do not fit or do not support this model on sm_120 with 24 GB.

## Running it

```bash
systemctl start gemma4-31b-fast     # 155K context, fast prefill (default on 8080)
systemctl start gemma4-31b-262k     # 262K context, QAT Q4, better knowledge scores (stops the fast unit)
```

Neither unit is enabled at boot, matching the other model units on CT116.

## Code and files

NInferno branch `kvshare-main` on CT116 (`/opt/NInferno`):

| Commit | Change |
|---|---|
| afb1bd1 | K-only KV for k_eq_v full layers, GeGLU split on prefill |
| 109084b | INT8 scores on the D512 FP8 prefill route |
| 739ea73 | `--activations int8` (W4A8) for Gemma 4 (opt-in, not used) |
| 1187685 | FP8 P·V on the D512 FP8 prefill route |
| 4e58bc6 | FP8 scores (promoted), replacing INT8 |

llama.cpp branch `qat-aware-q4_0` (`/opt/llama.cpp`, 3bd2f36) has the QAT-aware Q4_0 encoder. The patches are in [`patches/`](../patches/).

Artifacts on CT116:

- `/opt/models/dl2/qat-nvfp4.r31.ninferno`: QAT-NVFP4 target, 17.8 GiB.
- `/opt/models/dl/.ninferno-cache/gemma-4-31B-it-qat-UD-Q4_K_XL.gemma4_published.ninferno`: QAT Q4 target, 16.1 GiB.
- `/opt/models/dl2/qat-assistant.q40qat.ninferno`: QAT drafter, lossless Q4_0.
- `/opt/models/dl2/heretic-qat-q40qat.ninferno`: uncensored QAT-aware Q4_0, 262K, 104 tok/s.

## Next steps worth doing

- **A QAT-exact fast prefill:** an INT8 MMA GEMM that reads Q4_0 codes and per-32 scales directly. It would give Q4's accuracy at an estimated 2.5–3K prefill.
- **Faster decode GEMMs:** the NVFP4 A16 verify GEMMs run at ~545 GB/s of the 896 GB/s bus, and the Q6 head at ~455 GB/s, so roughly 10–20% of decode is left there.
- **Fused GELU·mul with NVFP4 activation quantization** on prefill: ~3%.
