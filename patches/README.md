# Patches

## NInferno (`ninferno/`, Apache-2.0)

Apply them in order with `git am` on [Brontir/NInferno](https://github.com/Brontir/NInferno) at `9895ebc62e2bdfb367083a72cb2c4a714cf41542` ([`BUILD.md`](../BUILD.md), step 1). Production runs all nine, at commit `e9203c5` on our branch `kvshare-main`.

| # | Patch | Active in production | What it does | Measured |
|---|---|---|---|---|
| 01 | `gemma4: K-only KV for k_eq_v full layers and GeGLU split on prefill` | yes, with both env switches | `NINFERNO_GEMMA4_SHARED_VALUES=1\|2`: the ten full-attention layers (D=512, k_eq_v, a constant key-norm weight c) have V = unrope(K)/c. Mode 1 keeps FP8 K plus V's 128 rotated dims; mode 2 keeps K only and turns the rotated pairs back with RoPE inside the attention kernel. `NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=N`: calls of N+ tokens run gate/up through the tuned linear route and apply GeGLU after it, instead of the fused epilogue (which ran the generic W4A4 kernel on NVFP4) | 262K context on 24 GB (Q4); NVFP4 prefill at 8K 2.5K → 3.9K tok/s with N=64; decode +6–9% with N=1 |
| 02 | `gemma4 attention: INT8 scores on the D512 FP8 prefill route` | code present, replaced by 05 | Q·K as m16n8k32 s8 MMAs on the FP8 prefill route of the 512-wide layers. 05 sets `kGemma4Fp8Scores`, which turns this path off | lab prefill 32K +8%, 64K +12%; perplexity 76.96 → 78.56. INT8 on the 256-wide sliding layers too was worse (85.05) |
| 03 | `gemma4: --activations int8 runs NVFP4 W4A8 prefill` | no (opt-in flag) | lets Gemma 4 take `--activations int8`: every recorded W4A4 projection is capped at 8-bit activations | NVFP4 W4A8 prefill runs at only ~900 tok/s here, and MMLU-Pro stays at 148: the NVFP4 quality gap is not in the activations |
| 04 | `gemma4 attention: FP8 P V on the D512 FP8 prefill route` | yes | P·V as m16n8k32 e4m3 MMAs straight from the landed V codes; P enters as E4M3 with its key's V scale and a gain folded in (2^13, 2^16 under shared values) | lab prefill 32K 2,992 → 3,522, 64K 2,034 → 2,542 tok/s; perplexity 78.56 → 77.99 |
| 05 | `gemma4 attention: FP8 scores on the D512 FP8 prefill route` | yes | Q·K as e4m3 MMAs on the K codes as stored, with Q in E4M3 against each row slice's absmax. The FP8 MMA accumulates in reduced precision, so each MMA starts from zero and is added to an FP32 sum | lab prefill 8K/32K/64K 4,986/3,522/2,542 → 5,110/3,748/2,786; perplexity 77.99 → 77.84. Without promotion, 81.13 |
| 06 | `nvfp4 shapes: four K warps on the Gemma 4 31B qkv eight-token tile` | yes | the QKV projections' A16 tile at T ≤ 8 runs 1×4 warps (as gate/up) instead of 2×2 | linear bench T=8: 16384×5376 81.9 → 71.0 µs, 18432×5376 90.1 → 73.7 µs; T=1 unchanged |
| 07 | `q6 head: the sliced 8-column tile for the Gemma 4 31B head at T 6..8` | yes | the tied Q6 head [262144, 5376] at T 6–8 (an MTP verify of up to 7 drafts) takes the sliced 8-column MMA tile instead of the 16-column one | linear bench T=8: 2.50 → 1.76 ms (442 → 627 GB/s) |
| 08 | `serve: finished-request sums and finish reasons on /metrics` | yes | `/metrics` adds, over completed requests since start, their count and summed seconds (e2e, queue, TTFT, prefill, decode), their prompt, cached and output tokens, speculative rounds, drafted and accepted tokens (`ninferno_request_*`, `ninferno_speculative_*`), and `ninferno_requests_finished_total{finish_reason}` (stop, length, tool_calls, abort, repetition; error for failed requests) | no speed change. A monitor differencing two samples lists every client's requests exactly (the NInfer web UI 0.1.0 Monitor does) |
| 09 | `program: a consumed endpoint is entitled only to the checkpoints it owns alone` | yes | a request that takes over a cached conversation (`PrivateEndpoint`, `ConsumeToActive`) counted every optional checkpoint of the conversation (rewrite state, long anchors) into its entitlement, also one another cached conversation still shares. The ownership count skips shared ones, so `start_request` threw `materialized sequence does not match its active entitlement` and the engine exited (expected `host_state 1`, actual 0). It now counts only checkpoints it owns alone, as the rewrite-restore path already did, and the error names the fields that differ | an agent loop that crashed the engine at the same step every time runs 93 steps, 89 tool calls, no error ([`bench/agenttest.py`](../bench/agenttest.py)) |

Patches 06 and 07 pay with 7 MTP drafts (`--draft-tokens 7`): the verify then runs 8 tokens.

## llama.cpp (`llama.cpp/`, MIT)

| Patch | What it does |
|---|---|
| `QAT-aware Q4_0: recover each block's BF16 QAT grid and store it exactly` | ggml's `quantize_q4_0` (with `GGML_Q4_0_QAT=1`) and gguf-py's Q4_0 (with `GGUF_Q4_0_QAT=1`) pick, per block, the k in 1..8 for which s = bf16(amax / k) puts every weight on an integer in [−8, 7]; blocks off the grid keep the reference encoder. `convert_hf_to_gguf.py --outtype q4_0` writes Q4_0 directly. On Google's QAT assistant, 100% of blocks are on the grid: 0.2% relative error against 5% for the reference encoder |

Apply it on [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) at `2280f0f` ([`BUILD.md`](../BUILD.md), step 2).

## Experiments (`experiments/`, not applied)

Env-gated changes that were measured and not adopted. They are diffs against `4e58bc6`.

| Patch | Switch | Result |
|---|---|---|
| `verify-a4-env.patch` | `NINFERNO_GEMMA4_VERIFY_A4=1`: the MTP verify keeps each projection's recorded W4A4 policy instead of A16 | chat decode 108.3 / 105.5 against 108.8 / 108.0 tok/s; MMLU-Pro 147. No gain |
| `q40-w4a8-ws-env.patch` | `NINFERNO_Q40_W4A8_WS=1`: GGML Q4_0 prefill on sm_120 takes NInferno's producer-warp W4A8 tile instead of ggml's MMQ | Q4 prefill +10% at 8K (1,644 → 1,806), +8% at 32K; perplexity 90.93 → 92.81 (+2%) on the quality profile |
