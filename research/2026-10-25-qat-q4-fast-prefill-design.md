# QAT Q4_0 fast prefill on the RTX 5090 Laptop (sm_120): which kernels run today, and a path to an INT8 W4A8 GEMM at 2.5K tok/s or more

Date: 2026-10-25. Source read: `/opt/NInferno` on CT116, branch `kvshare-main` at `4e58bc6`.
`src/models/gemma4/target.cu` has an uncommitted change on the box, which nothing below depends on.
Logs and data come from `/root/gbench`. Nothing was built, and nothing ran on the GPU. The only
computation was one CPU-only numpy read of the GGUF's Q4_0 scale planes.

Tags: **[V]** means verified by reading code, logs or data. **[I]** means inferred or estimated and
needs a measurement.

---

## TL;DR

1. **The current prefill kernel for both Q4 artifacts is ggml's `mul_mat_q` (Family::GgmlMmq), not
   NInferno's own W4A8 tiles.** The heuristic gives ggml's int8 MMQ to every ggml block format
   from 17 tokens up, in 512-token slices, whenever `mmq_prefill_wins()` holds. On this laptop it
   holds because the nearest measured board is the RTX 5090, whose gate is `true`. The route-table
   and tuner paths cannot override it for A8 calls. [V] in code; the board-distance outcome is [I]
   and depends on the L2 size the driver reports (see §1.4). Both artifacts are 100 % `GGML_Q4_0`,
   including the one called "UD-Q4_K_XL", so Q4_K never runs in them. [V]
2. **NInferno already has an INT8 `mma.sync m16n8k32` W4A8 kernel that reads GGML Q4_0 planes as
   stored**: the classic tile, and a producer-warp tile `w4a8_ws_kernel`, both compiled for Q4_0.
   Neither runs on sm_120, for two reasons. MMQ wins the route first. And the producer-warp tile is
   turned on for Q4_0 only on Ada (`w4a8_ws_wins`). [V] No format conversion is needed to feed Q4_0
   to it.
3. **No existing format holds QAT Q4_0 losslessly *and* has a faster prefill route.** `U4_G32_ZP_FP16`
   is lossless (code = q, zero = 8, scale = d) but runs on the same W4A8 classic tiles and costs
   +0.25 bit per weight. `Q8_G32` is lossless but needs about 29 GiB. The fast I8 route (0.53 of int8
   peak) takes only `Q4/Q5/Q6_G64`, which has one scale per 64 values. In the QAT checkpoint the two
   32-blocks of a 64-group share a scale in only about 0.3 % of groups, so that conversion is lossy. [V]
4. **Plan, smallest first:**
   - (0) Use nsys to confirm the kernel mix.
   - (1) A switch of about 2 lines that sends Q4_0 prefill on sm_120 to the existing producer-warp
     W4A8 tile instead of MMQ, then an A/B. The outcome is uncertain, roughly −10 % to +30 %.
   - (2) A Q4_0-specialised INT8 kernel cloned from `generic_i8.cu`. It feeds raw Q4_0 code words
     straight into the MMA as the u8 B fragment, with no decode and no permutation, since GGML's
     nibble order already *is* the m16n8k32 B-fragment order. The zero point goes into the
     accumulator init, it uses a cp.async or TMA ring, and it applies a lean per-32 epilogue.
   - (3) If INT8 stalls below about 0.6 of peak, an MXFP8-activation variant. It uses the existing
     block-scaled `kind::mxf8f6f4` tile, where the hardware applies the per-32 activation scale, so
     only one FFMA per accumulator element remains.
5. **What 2.5K tok/s requires** (6.5K-token prompt, §4.1): the Q4 GEMMs must sustain about **0.62 of
   the INT8 peak**. Today they reach about 0.34. 3K tok/s needs about 0.8, or less non-GEMM time.
   This is reachable in principle, but the per-32 software epilogue uses 9–12 of the 16 issue slots
   that each INT8 MMA has on sm_120 (§4.2). That is the core difficulty, and it is why (3) exists.

---

## 0. Facts about the weights and the call path

### 0.1 Artifact contents [V]
- `/opt/models/dl/.ninferno-cache/gemma-4-31B-it-qat-UD-Q4_K_XL.gemma4_published.ninferno.conversion.json`
  and `/opt/models/dl2/heretic-qat-q40qat.ninferno.conversion.json` both list
  `formats: {ggml_q4_0: 241, bf16: 361, fp32: 60}`. Every matrix is `ggml_q4_0`, layout
  `ggml_planes_v1`, kept with `import_encoded`. That covers q/k/v [16384 or 18432 × 5376], o, the
  gate+up pair [43008 × 5376], down, and the token embedding.
- The source GGUF `/opt/models/dl/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf` holds tensor types
  `{Q4_0: 411, F32: 422}`. There is no Q4_K anywhere, despite the name.
- `GgmlPlanes` layout for Q4_0: a code plane of 16 B per block per row, then a scale plane of one
  fp16 `d` per block (`src/core/weight.h:111`, `:164`, layout comment `:152-163`). Q4_0 semantics:
  `w = d*(q-8)` (`weight.h:56-60`).

### 0.2 Structure of the QAT scales
This is a CPU numpy read of the scale planes of `blk.0.ffn_gate`, `blk.0.ffn_down`,
`blk.30.attn_q` and `blk.59.ffn_up`, 3.6 M blocks each. [V]

| property | value | consequence |
|---|---|---|
| `d` exactly a power of two | 0.05 % | MX (UE8M0) block scales cannot carry `d` |
| `d` mantissa fits 3 bits (E4M3/UE4M3) | 0.13 % | NVFP4-style UE4M3 scales cannot carry `d` |
| `d` mantissa fits bf16 | about 7 % | arbitrary fp16 scales |
| 64-groups whose two 32-blocks share `d` | 0.26–0.30 % | Q4_0 → Q4_G64 (the I8 route's format) is lossy |
| code range | −8..7, with −8 used about 2.5 % of the time | E2M3 (max 7.5) is out; int8, E4M3 and E3M2 are exact |
| negative `d` | 0.4–0.6 % of blocks | any unsigned-scale format must move the sign into the codes |

So any exact path must apply an arbitrary fp16 scale per (row, 32-block) in software. No tensor-core
block-scale format (UE8M0, UE4M3) can hold it.

### 0.3 Policy [V]
- `kept_block_policy(GGML_Q4_0) = AllowA8` (`src/convert/formats.cpp:1070-1105`).
- `ActivationMode::Fast` is the default (`include/ninferno/types.h:137-148`) and keeps the recorded
  policy (`src/models/common/weight_use.h:25-30`).
- The logs confirm it: the dense-tune keys read `gb203:82sm|q40|a8|...`
  (`/root/gbench/logs/P4-densetune-chunk2048.log`).

### 0.4 Call path [V]
- Gemma 4 projections go through `project()` → `ops::linear` (`src/models/gemma4/target.cu:101-106`).
- Gate+up goes through `ops::linear_geglu` (`target.cu:496-506`). With
  `NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1`, which both deployed units set, it runs as a Store GEMM plus
  `gelu_tanh_mul` instead (`target.cu:507-511`).
- `dispatch_linear` (`src/ops/linear/linear.cpp:171-201`) sends GGML formats to the generic path,
  because `linear_shape_registered()` has no ggml entries (`linear.cpp:27-60`).
- From there: `generic::run` → `select_route` (`src/ops/generic/generic_gemm.cpp:3709-3723`).

### 0.5 Work per token [V]
50 sliding layers have q+k+v = 16384 rows; 10 full layers have q+k = 18432 rows (k_eq_v). Each layer
also has o (8192 or 16384 columns), gate+up 43008 × 5376 and down 5376 × 21504.

Total: **29.29 G MACs, which is 58.57 GFLOP per token** in the body linears. The LM head runs only on
the last token.

---

## 1. Question 1: prefill route selection for Q4_0 and Q4_K on sm_120

### 1.1 Route families
These are in `enum class Family` (`src/ops/generic/generic_gemm.h:174-194`):

| Family | What it is |
|---|---|
| `RegisterMma` | A16 |
| `Scalar` | fallback |
| `W4a4Mma` | NVFP4 only |
| `TiledMma` | A16 shared-memory tiles |
| `W4a8Mma` | `generic_w4a8.cuh`: int8 m16n8k32, per-32 activation scale |
| `StagedMma` | BF16 |
| `Nvfp4TiledMma` | NVFP4 tiled |
| `R8Mma` | decode, T ≤ 8 |
| `GgmlMmq` | ggml `mul_mat_q`, stream-k |
| `GgmlVec` | dp4a, 1–8 tokens |
| `I8Mma` | `generic_i8.cu`, Q4/Q5/Q6 **G64 only** |

### 1.2 The decision for a prefill call of T = 512..2048 [V]
1. `select_route` (`generic_gemm.cpp:2636-2741`) sets `a8_wide = tokens >= 17 && allows_w4a8(...)`
   (`:2702`). Q4_0 passes `allows_w4a8`: it is in `w4a8_supports` (`:1487-1528`), K is a whole
   number of tiles (`:1462-1485`), and the policy is AllowA8. With `a8` set, the route tables are not
   consulted (`:2706`), nor is the dense tuner's decode route (`:2720`).
2. `heuristic_route` (`:2523-2611`), with `prefill_band = true`, steps through these checks:
   - W4A4: no, the weight is not NVFP4 (`:2529`).
   - vec rule: no, that is for T ≤ 4 (`:2542`).
   - cost model: skipped under the prefill profile (`:2545`).
   - R8: no (`:2548`).
   - **`if (tokens >= 17 && mmq_prefill_wins(problem, weight)) return Route{Family::GgmlMmq, 0};`
     (`:2552`).** This is the return that fires.
   - Only after that come `i8_route` (`:2553-2555`, G64 only) and `w4a8_route` (`:2556-2558`).
3. `mmq_prefill_wins` (`:503-506`) holds when all of these do:
   - `mmq_prefill_device()` returns true for the nearest measured board (`:494-501`).
   - The format is an MMQ format: Q4_0 at `:403-404`, Q4_K at `:391-392`.
   - n % 128 == 0. All Gemma shapes pass: 16384, 18432, 5376 = 42 × 128, 43008.
   - `mmq_admits` (`:472-479`): K % 256 == 0 (5376, 8192, 16384, 21504 all pass), GgmlPlanes for
     Q4_0 or GgmlBlocks for Q4_K, and 16-byte aligned planes (`:459-470`).
4. A tuned prefill band cannot replace GgmlMmq. `tuned_prefill_route` only returns a band whose base
   equals the heuristic (`src/ops/generic/generic_tune.cpp:749-762`). For a GgmlMmq base,
   `prefill_candidates` is empty, because `same_summation` is false for every candidate
   (`generic_tune.cpp:282-303`). No route table holds a `q40` entry: gb203, gb202, gb206 and ad102
   were all checked.
5. Execution (`run_route`, `generic_gemm.cpp:3373-3405`):
   - Per 512-token slice (`kMmqPrefillSlice`, `:456`): `ggml_mmq::quantize_activation` (q8_1), then
     `ggml_mmq::product` into an **FP32 plane** of `rows × 512` (88 MB for gate+up), then a separate
     `launch_mmq_epilogue` kernel to BF16.
   - ggml's config for Q4_0 on Blackwell falls through to the Ampere table
     (`third_party/ggml-quants/src/ggml-cuda/mmq-config-blackwell.cuh:36`, `mmq.cuh:257-262`):
     256 threads, occupancy 1, a 128-row × up-to-128-token tile, int8 SRAM layout `Q8_0`, stream-k
     (`mmq-config-ampere.cuh:36-51`).
   - The Q4_0 unpack to int8 and the per-32 `d_w*d_a*(float)sumi` epilogue are ggml's.

**Q4_K**: the same path, GgmlMmq. If MMQ were off, Q4_K would take the producer-warp W4A8 tile on
Blackwell (`w4a8_ws_wins`, `:1021-1022`). [V]

### 1.3 Why MMQ "wins"
The gate is a per-board measured choice, not a per-format one (comment at `:481-493`).
- It was measured with ISTA IQ3_S and an "Unsloth UD-Q4_K_XL" mix on the 4060 Ti, L4, A10,
  2080 Ti, 4090 and 5090. On the 5090: 3930 → 4306 and 3771 → 4152 tok/s.
- The formats that drove it are the classic-tile ones. For Q4_0, MMQ against the W4A8 producer-warp
  tile **was never measured on any card**.
- The only Q4_0 data in the code is the Ada rule (`:958-965`). On a 4090, Gemma 4 31B QAT prefill
  went from 1914 to 2149 tok/s with the producer-warp tile against the W4A8 64x64 tile. Per call,
  gate+up took 0.51× the time. That rule is now shadowed on the 4090 too, because the 4090's MMQ
  gate is `true` (`:495`). [V] for the code; the measurement order cannot be recovered, since the
  history is one squashed import.

### 1.4 The board-distance caveat [I]
`board_distance` is |log2 SM ratio| + |log2 bandwidth ratio| + 0.5·|log2 L2 ratio|
(`src/core/device_class.cpp:415-430`, `src/core/measured_boards.h:15-17`). The laptop has 82 SMs and
896 GB/s, as the engine logs.

| L2 the driver reports | distance to 5090 | distance to 5060 Ti | nearest | route |
|---|---|---|---|---|
| 64 MiB (expected for GB203) | 2.34 | 2.69 | 5090 | MMQ |
| 48 MiB | 2.55 | 2.48 | 5060 Ti | W4A8 64x64 classic tile |

Supporting [V]: the dense tuner did pick `ggml_mmq` at T 5–16 on this card (P4 log), so MMQ
admission holds here. **Verify the prefill kernel with nsys before changing anything (step 0).**

### 1.5 Side observation [I]
`route_is_token_invariant(GgmlMmq) == false`, because stream-k splits K by tile count
(`generic_gemm.cpp:2144-2145`). Yet the heuristic returns it under the prefill profile, whose header
promises chunk-size-invariant bytes (`include/ninferno/ops/prefill_routes.h`). The W4A8 and I8
routes are token-invariant (`:2134-2149`). Moving Q4_0 off MMQ therefore also restores that
invariance.

### 1.6 Measured speeds on this box [V]
Source: `/root/gbench/results.jsonl`. "pp8192" is a 6573-token prompt.

| Configuration | pp8192 tok/s |
|---|---|
| QAT-UD Q4_0, chunk 512–4096 (P1–P3, Q0, Q1) | 1518–1571, independent of chunk size, as expected for 512-token MMQ slices |
| QAT-NVFP4 W4A4 with GeGLU split (UF / D1) | 4247–4254 |
| QAT-NVFP4 under `--activations int8` (P8): generic W4A8, **64x64 classic tile**, NVFP4 `kHalves` codec | **907**. A data point for the classic W4A8 tile on this card, though with a codec that needs 2× the MMAs and 2× the epilogue of Q4_0 |
| llama.cpp, the same QAT-UD GGUF (L1, 7471 tokens) | 1334 |

---

## 2. Question 2: does an existing INT8 W4A8 route take a Q4_0-like g32 layout?

**Yes. The generic W4A8 reads GGML Q4_0 planes natively.** [V]
- Codec `W4a8Ggml32<Ggml32::Q4_0>` (`src/ops/generic/generic_w4a8.cuh:215-260`) loads 16 code bytes
  and the fp16 `d` per block straight from the GgmlPlanes planes (`:225-232`). It decodes to int8 in
  shared memory, `nibble − 8` (`:244-255`).
- It launches from `generic_w4a8_ggml32.cu:11-13`.
- Arithmetic: `mma.sync m16n8k32 s8.s8.s32` per 32-block, then
  `o += I2FP(c) * (as*ws)` for each of the 4 accumulators (`generic_w4a8.cuh:1586-1593`). That is
  4 I2FP + 4 FMUL + 4 FFMA per MMA. I2FP is used on sm_80+ because the biased-float trick measured
  slower on Ada and Blackwell (`src/ops/common/mma.cuh:9-36`).
- Activation quantizer: int8 with an fp32 scale per 32 (`generic_w4a8.cu:15-41`).
- Tiles (`generic_gemm.cpp:147-154`, `generic_w4a8.cuh:1896-1902`): 32x128, 64x128, 128x128, 128x64,
  64x64, and **128x64 with 4 producer warps** (`w4a8_ws_kernel`, `:1823-1885`, 3-stage ring,
  `w4a8_mma_tile` reading the scales as float4).
- The producer-warp tile **is compiled for Q4_0** (`w4a8_ws_compiled`, `:1909-1916`) and admitted
  (`w4a8_ws_supports`, `generic_gemm.cpp:292-296`).
- On sm_120, `heuristic_w4a8` (`:1060-1116`) gives Q4_0 **64x64** from 128 tokens. Q4_0 is not a
  "wide format", and `w4a8_ws_wins` returns `ada && n>=5376 && k>=5376` for Q4_0 (`:1023-1024`), so
  it is false on Blackwell.
- The W4A8 workspace is already reserved for every A8 problem, alongside MMQ's
  (`generic_gemm.cpp:3149-3150`).

**Feeding Q4_0 QAT weights to it therefore needs no conversion and no new decoder. It needs only
route selection** (step 1 in §4).

The other INT8 route, **I8** (`generic_i8.cu`), takes only `Q4/Q5/Q6_G64_FP16`:
- `allows_i8`, `generic_gemm.cpp:281-287`; `launch_epilogue` throws otherwise (`generic_i8.cu:378-388`).
- It is the most efficient int8 kernel in the tree: **0.49–0.57 of measured int8 peak** on the
  3090, 4090 and 5070 Ti (`generic_i8.cu:20-23`, `kI8Eff` at `generic_gemm.cpp:2207-2209`).
- It gets there by feeding the 4-bit code to the MMA as the unsigned operand, which avoids a decode
  pass, by folding the zero point and the float bias into the accumulator init, and by sharing one
  scale per 64 values. Two MMAs share one epilogue: 6 instructions per MMA.

---

## 3. Question 3: an existing int4 + per-32-scale format with a faster route?

| Target format | Lossless from QAT Q4_0? | Size | Prefill route on sm_120 | Verdict |
|---|---|---|---|---|
| `GGML_Q4_0` (as is) | – | 4.5 bpw, 16.3 GiB | MMQ today; W4A8 tiles available | baseline |
| `U4_G32_ZP_FP16` (40) | **yes**: code = q, zero = 8, scale = d. Negative d is fine in fp16. Needs a nibble repack from GGML split-half order to natural order (`weight.h:27-31`, layout `RowGroupZp` `:150`) | 4.75 bpw, about +0.9 GiB | W4A8 classic tiles only (`W4a8U4ZpG<32>`, `generic_w4a8_superblock.cu:39-41`). Not producer-warp compiled, no MMQ, no I8 | no gain, costs memory |
| `GGML_Q4_1` | yes (m = −8d is exact in fp16) | 5.0 bpw | MMQ / W4A8 with a min term (+4 FFMA/MMA) | worse |
| `Q8_G32_FP16` (3) | yes | 8.5 bpw, about 29 GiB | – | does not fit 24 GB |
| `Q4_G64_FP16` (0) | **no**: only 0.3 % of 64-groups share d | 4.25 bpw | **I8** (0.53 of peak) | lossy, defeats QAT |
| `MXFP4` / `NVFP4` | no (E2M1 has no ±5, ±7; scales are UE8M0/UE4M3) | – | – | – |
| `IQ4_NL` | no (non-uniform codebook) | – | – | – |

**Conclusion [V]: no lossless conversion buys a faster route. The cheapest wins are route selection
(step 1) and a Q4_0-specialised I8-style kernel (step 2), which keeps the artifact unchanged.**

---

## 4. Question 4: recommended plan

### 4.1 Target arithmetic
- **NVFP4 reference (kernel-time split) [V]**: `prof_nv_split` is a 6656-token prompt run 4 times.
  - GEMM 3.203 s, attention 1.978 s, other 0.650 s.
  - Per run that is GEMM 0.80 s and non-GEMM 0.66 s, for 1.457 s total and 4568 tok/s.
  - GEMM throughput = 6656 × 58.57 GFLOP / 0.80 s ≈ **487 TOPS**.
- **INT8 peak [I]**: take the NVFP4 GEMM to be 77 % of FP4 peak (your figure). Then FP4 peak is about
  632 TOPS, and **INT8 peak is about 316 TOPS**, half the FP4 rate (`s8 = 2048` ops/clk/SM,
  `src/core/generation.h:56`). That implies a sustained clock of about 1.88 GHz. At the 1597 MHz
  reference clock it would be 268 TOPS.
- **Q4_0 today [I]**: 6573 tokens at about 1560 tok/s is 4.21 s. With the same about 0.66 s of
  non-GEMM time, the GEMM time is 3.55 s, which is about 109 TOPS, or **about 0.34 of INT8 peak**.
- Model: `tok/s = 6573 / (0.66 + 385 TFLOP / (316 TOPS × e))`, where e is the GEMM's fraction of
  INT8 peak, quantiser included.

| e (fraction of INT8 peak) | 0.34 (today) | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.77 |
|---|---|---|---|---|---|---|---|---|
| tok/s at 6.5K | about 1550 | 1950 | 2120 | 2290 | 2440 | 2590 | 2740 | 2930 |

So 2.5K tok/s needs e ≈ 0.62. 3K needs e ≈ 0.8, or a cut to attention and the other non-GEMM work.
That is about 80 % of the relative efficiency the NVFP4 TMA kernel achieves.

### 4.2 Why per-32 INT8 is hard on sm_120 [I]
- At the full INT8 rate (1024 MAC/clk/SM), each SM sub-partition finishes one m16n8k32 (4096 MAC)
  every **16 cycles**, and it issues about 1 warp-instruction per cycle. That leaves about 15 issue
  slots per MMA for everything else.
- Per-MMA epilogue cost, per 4 accumulator elements:

| Design | Instructions per MMA | Notes |
|---|---|---|
| current W4A8 (`generic_w4a8.cuh:1590-1593`) | 4 I2FP + 4 FMUL + 4 FFMA = **12** | plus the per-CTA weight decode into smem: about 1–2 per MMA at a 128-token tile, using `__vsub4`, which is "four instructions" (`:99`) |
| I8 g64 (`generic_i8.cu:301-309`) | 3 per element per **2** MMAs = **6** | – |
| I8-style **g32**, exact per-32 activation scale | **12** | the same FADD + FMUL + FFMA, now per MMA |
| g32 with a **per-128 activation scale** and a per-K-tile partial accumulator | about **9** | doubles the accumulator registers, so use 64×32 warp tiles |
| **MXFP8 activations** through block-scaled `kind::mxf8f6f4` | **4** FFMA (acc += d_w × fresh D) | the hardware applies the per-32 UE8M0 activation scale; the weight `(q−8)` is exact in E4M3, with the 16-entry nibble LUT decode done by producer warps (about 1.5 per MMA) |

  On sm_120 the block-scaled form also avoids the plain FP8 form's half-rate FP32 accumulation
  (`mma.cuh:486-512`). [V for that note]
- Add A/B fragment loads (about 0.25–1 per MMA) and the MMA itself. The exact-INT8-per-32 design
  then sits at about 14–15 of 16 slots, so expect **e ≈ 0.40–0.50**. Per-128 activation scales give
  about 11 of 16, so **e ≈ 0.55–0.65**. MXFP8 gives about 7 of 16, so it can be MMA-bound like the
  NVFP4 kernel: **e ≈ 0.65–0.75**, but with E4M3 activation precision.

### 4.3 Steps

**Step 0: confirm the kernel mix (measurement only, no code).**
- Profile one 6.5K-token prompt on the Q4_0 artifact, as in `/root/gbench/prof_nv.sh`.
- Expect `mul_mat_q<(ggml_type)2,...>`, the ggml q8_1 quantiser, and NInferno's `mmq_epilogue`
  kernels to dominate.
- Read per-GEMM time for gate+up, qkv, o and down, and compute the achieved TOPS. This pins down
  "e today" and settles the L2 caveat in §1.4.

**Step 1: route Q4_0 prefill to the existing producer-warp W4A8 tile (about 2 lines; needs only a rebuild).**
- `src/ops/generic/generic_gemm.cpp:1023-1024` (`w4a8_ws_wins`):
  `case QType::GGML_Q4_0: return (ada || blackwell) && p.n >= 5376 && p.k >= 5376;`
- `src/ops/generic/generic_gemm.cpp:503-506` (`mmq_prefill_wins`): return false when
  `p.qtype == QType::GGML_Q4_0 && w4a8_ws_wins(p, w)`. This needs a forward declaration, since
  `w4a8_ws_wins` is defined at `:996`. `ggml_mmq_gated_pair` follows automatically, because it asks
  `select_route` (`:2834-2843`).
- Nothing else changes:
  - The tile is compiled for Q4_0 (`generic_w4a8.cuh:1913`).
  - The workspace is already sized (`generic_gemm.cpp:3149-3150`).
  - The token-major grid is already on (`:681`).
- A/B pp8192 and pp32768 with gbench, both with `NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1` and unset. The
  W4A8 SwiGlu epilogue fuses GeGLU, so the unset arm saves the `gelu_tanh_mul` pass.
- Expected **[I], uncertain**: anywhere from −10 % to +30 % (about 1.4–2.0K tok/s).
  - For: on the 4090 this tile halved gate+up time against the 64x64 tile (`:958-962`), and the
    route cost model rates it at about 2.2× MMQ's efficiency (`kW4a8Eff[5]=0.98` against MMQ `0.44`,
    `:2338`, `:2395`). That model is not used under the prefill profile.
  - Against: on the 5090 MMQ beat the W4A8 tiles for Q4_K mixes (`:487-488`).
- If Step 1 wins, keep it. Either way it gives a calibrated W4A8 number for Step 2.

**Step 2: a Q4_0-specialised INT8 kernel, "I8-g32" (main recommendation for exact QAT).**
Clone `generic_i8.cu` and change only what g32 needs. Everything below about B fragments and
alignment is [V]-consistent with the code.

- **B fragment straight from the stored code bytes.**
  - GGML Q4_0 stores value j in the low nibble of byte j and value j+16 in the high nibble
    (`generic_w4a8.cuh:128-129`).
  - The m16n8k32 B fragment wants k = 4t..4t+3 in b0 and k = 16+4t..16+4t+3 in b1. The I8 kernel
    relies on this same mapping (`generic_i8.cu:13-16`, `:276-281`).
  - So lane (n = g, t) needs **word t of the block's 16 code bytes**:
    `b0 = w & 0x0F0F0F0F`, `b1 = (w >> 4) & 0x0F0F0F0F`.
  - That means no decode, no XOR (Q4_0 codes are already offset-binary with zero 8), and no
    activation permutation. The activation stays in natural k order, so the existing
    `w4a8_quantize_block` (`generic_w4a8.cuh:59`) can be reused.
  - An `ldmatrix.x4` over four n8×16 B row-tiles in shared memory returns exactly these words for
    4 n-tiles in one instruction.
- **MMA** `mma.sync.m16n8k32.row.col.s32.s8.u8.s32` (as `i8_mma`, `generic_i8.cu:81-91`).
  - C-init = `0x4B400000 − 8·Σq_x[t,b]`, folding the zero point and the float bias as I8 does
    (`:263-264`). The quantiser must emit the int code sum per (token, 32-block), or this init
    directly.
  - Pass C and D as separate registers to avoid the per-MMA copies.
- **Staging.**
  - Copy code rows (16 B per block) and fp16 scales (16 B = 8 blocks per row: an 8-block ring as
    I8's `:222-226`) straight from the GgmlPlanes planes with `cp.async` or TMA.
  - Row strides are 16-B aligned for every Gemma K: bpr × 16 B for codes, bpr × 2 B = 336/512/1024/1344 B
    for scales.
  - Use warp-specialised producers and 128–256-value K stages to amortise barriers. The template is
    the NVFP4 TMA schedule: 256 × 128 CTA, 8 consumer warps of 64×64, 3 stages
    (`src/ops/linear/nvfp4/nvfp4_schedule.cuh:134-164`).
- **Epilogue: two variants behind one template flag.**
  - (a) Exact per-32 activation scale, the same numerics class as today's W4A8.
  - (b) Per-128 activation scale with a partial accumulator, about 25 % fewer issue slots, costing
    some activation precision. Measure it with qdetail/geval as for the NVFP4 work.
- **Plumbing.**
  - `allows_i8` (`generic_gemm.cpp:281-287`), `i8_bits` (`:289`), `i8_fits` (`:2211-2220`).
  - `route_admits` I8 case (`:2113-2115`).
  - The I8 workspace in `route_workspace_bytes` (`:2994-3002`, groups of 32) and in
    `workspace_capacity_bytes` (`:3144-3148`: add `GGML_Q4_0`).
  - In `heuristic_route`, place `i8_route` for Q4_0 before `mmq_prefill_wins` (`:2552-2555`), or
    exclude Q4_0 from MMQ as in Step 1.
  - `launch_i8` / `launch_epilogue` (`generic_i8.cu:378-413`).
  - The route stays token-invariant (`:2148`), which also fixes §1.5.
- Expected **[I]**: e ≈ 0.45–0.55 with (a), so **about 1.95–2.3K tok/s**. With (b) e ≈ 0.55–0.65,
  so **about 2.3–2.6K**. The memory format and QAT weight exactness are unchanged.

**Step 3: an MXFP8-activation variant for the last stretch toward 2.6–2.9K tok/s.**
Exact weights, but E4M3 activations.

- Reuse what already exists:
  - The block-scaled tile `w4a8_mma_tile_mx` (`generic_w4a8.cuh:1615-1666`, sm_120a only).
  - The MXFP8 activation quantiser `launch_moe_mx8_quantize`
    (`src/ops/moe/moe_w4a8_quantize.cu:30-72`; E4M3 codes with a power-of-two scale per 32).
- New codec: decode `(q−8)` to E4M3 bytes in the producer warps, which is exact (§0.2), using a
  16-entry LUT through `__byte_perm` like `iq4nl_s8` (`generic_w4a8.cuh:116-126`). Use a unit
  UE8M0 weight scale (0x7F).
- New tile variant: each k32 MMA writes a zeroed fragment, then `acc += d_w[n,b] * D` (4 FFMA).
- Accuracy is the open question. MXFP8 keeps about 4 significant bits per activation, against int8
  per-32 absmax's 7 bits relative to the block max. Gate it on the MMLU-Pro 280-question subset and
  KLD against BF16, since the reason to run Q4 QAT is that 164-of-280 score.

### 4.4 Smaller items worth folding in
- With MMQ, every 512-token slice writes and reads an FP32 plane: `rows × 512 × 4` B, which is
  88 MB for gate+up. It also runs a separate epilogue kernel (`generic_gemm.cpp:3379-3402`). The
  W4A8 and I8 routes write BF16 straight from registers.
- The int8 activation quantiser is a separate kernel per GEMM (`generic_w4a8.cu:46-55`), about
  1–2 % of prefill [I]. It could later be fused into `gemma4_residual_norm`, which already emits
  the FFN input.
- For the 5376-row GEMMs (o, down) at T = 2048, a 128 × 128 tile gives 672 CTAs, which is 8.2 waves
  on 82 SMs. Use the NVFP4 kernel's rasterisation so weights and activations stay L2-resident:
  the down-projection activation is 44 MB against about 64 MB of L2.

---

## 5. What was verified and what is assumed

- **Verified [V]**:
  - Route code paths and line numbers; ggml config; codecs; kernel arithmetic.
  - Artifact formats; GGUF tensor types; scale statistics.
  - Bench numbers and the NVFP4 profile split.
  - That MMQ runs on this card at T 5–16 (tuner log).
- **Not verified, needs a GPU run**:
  - Which prefill kernel actually runs on this laptop (depends on the L2 reported, §1.4).
  - The real per-GEMM efficiency of MMQ today: 0.34 is derived from end-to-end numbers.
  - The producer-warp W4A8 tile's speed for Q4_0 on sm_120.
  - Whether sm_120 has FP32x2 FFMA2. If `fma.rn.f32x2` compiles to FFMA2 on sm_120a, the FP32
    epilogue cost in §4.2 halves. Check it with ptxas or cuobjdump.
  - All efficiency projections in §4.
