# Fix NInferno's drafter before switching engines

The upside on this laptop lies inside NInferno, not in another engine: nothing in the published record beats the current setup across the board on a single 24 GB RTX 5090 Laptop GPU. Its K=V-deduplicated FP8 global cache is the only single-GPU route to 262K tokens the research found (mainline llama.cpp stores K and V separately and stops near 128K at q8_0). Its ~37 tok/s non-speculative decode already pulls ~71% of the 896 GB/s bus, and its ~1.45K tok/s prefill sits inside the band llama.cpp-class kernels would reach on this chip. The cheapest large win is a defect hunt. At depth 3 the MTP drafter lands 2.1 tokens/round on code, while llama.cpp and ik_llama.cpp report roughly 2.9–3.2 on mixed prompts and vLLM finds code is the assistant's *best* category. That gap points to a fixable mismatch in KV visibility, the vocabulary shortlist, QAT pairing or drafter inputs, worth an estimated 20–40% on code and math decode. Beyond that, three code projects carry the real upside:

- **Lower-precision attention for the ten head_dim-512 global layers.** Past ~180K tokens these layers outweigh all of the model's GEMMs, and today they run at GeForce's halved FP32-accumulate rate.
- **A 4-bit Hadamard-rotated K-only cache.** It frees ~2.2 GiB at 262K for about one MRCR point of quality.
- **A DFlash drafter routed to code and math.** It has delivered 2.4–3.8x on 24 GB cards.

Weight quality is already at the 16 GiB frontier: only pajari's Q4_0_V2 matches Unsloth's QAT UD-Q4_K_XL. The quality work is therefore mostly about not losing that quality to KV or attention shortcuts. vLLM, SGLang, TensorRT-LLM and ExLlamaV3 are not viable replacements on this card. Mainline llama.cpp is still worth one afternoon as a reference implementation for MTP acceptance and NVFP4 prefill.

## No off-the-shelf engine beats NInferno on a 24 GB laptop

Gemma 4 31B runs in the llama.cpp family, ExLlamaV3, vLLM, SGLang and TensorRT-LLM, but only the first two fit a single 24 GB card in practice.

- **vLLM:** the NVFP4 build reaches only **about 25K tokens of context on a 32 GB desktop RTX 5090** ([LilaRest NVFP4-turbo](https://huggingface.co/LilaRest/gemma-4-31B-it-NVFP4-turbo)). Forcing FlashInfer on sm_120 fails because the kernel rejects the 512-wide global heads ([vLLM #40677](https://github.com/vllm-project/vllm/issues/40677)).
- **SGLang:** the cookbook requires `--tp 2` for 31B with MTP ([SGLang cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/Google/Gemma4)).
- **TensorRT-LLM:** a maintainer confirmed that Gemma NVFP4 is unsupported on SM120 ([TRT-LLM #17052](https://github.com/NVIDIA/TensorRT-LLM/issues/17052)).
- **ExLlamaV3:** it supports the model with DFlash, n-gram drafting and 2–8-bit KV, but Gemma 4 MTP support is still an open request ([exllamav3 #567](https://github.com/turboderp-org/exllamav3/issues/567)).

Context is where NInferno's lead is structural. The ten global layers carry 4 KV heads × 512 dims with `attention_k_eq_v: true` ([config.json](https://huggingface.co/google/gemma-4-31B-it/blob/main/config.json)). Storing the shared tensor once costs 40 KiB/token at FP16 and ~20 KiB at FP8. Atomic Chat's 4090 VRAM deltas (2,591 MiB per extra 32,768 tokens at FP16, about 81 KiB/token) imply that mainline llama.cpp stores K and V separately. That caps it at **128K with q8_0 KV, using 23,725 MiB on a 24 GB card** ([Atomic Chat 4090](https://atomic.chat/blog/guides/best-local-llms-for-rtx-4090)). The other forks fall short too:

- **beellama:** reaches 200K at q5_0/q4_1 KV only at the edge of memory, and runs out of memory at 262K ([club-3090 #239](https://github.com/noonghunna/club-3090/discussions/239)).
- **ik_llama.cpp:** hits a ~24K wall with its full-length SWA cache ([club-3090 #239](https://github.com/noonghunna/club-3090/discussions/239)), and its `--swa-compress` fix is still an open PR ([ik_llama PRs](https://github.com/ikawrakow/ik_llama.cpp/pulls)).

No source shows another single-GPU engine exploiting K=V.

On speed, llama.cpp is a peer, not a successor. On a desktop RTX 5090 at build b10988, Gemma 4 31B Q4_K_M ran **3,698 tok/s prefill and 68.5 tok/s decode, while CISCai's NVFP4 GGUF ran 5,826 and 69.7** ([Atomic Chat 5090](https://atomic.chat/blog/guides/best-local-llms-for-rtx-5090)). Scaled by the laptop's compute and bandwidth, that gives:

| llama.cpp on the laptop (projected, not measured) | Estimate |
|---|---|
| Prefill, Q4_K | ~1.1–1.8K tok/s |
| Prefill, NVFP4 | ~1.7–2.9K tok/s |
| Decode, no speculation | 36–43 tok/s |
| Decode with MTP, short context | 58–80 tok/s |

The MTP row applies llama.cpp's measured speedup: 1.55x for a Q4 target at n=3 on an RTX PRO 6000 ([llama.cpp PR #23398](https://github.com/ggml-org/llama.cpp/pull/23398)).

These projections carry two caveats:

- **No laptop measurement exists.** There is no Gemma 4 31B benchmark for the 5090 Laptop, and the same PR records one user at ~14.6 tok/s with q8_0 KV against 50.7 with FP16 KV, even after the MTP fix.
- **The build flag matters.** Without `-DGGML_CUDA_FA_ALL_QUANTS=ON`, a 3090 ran ~12 tok/s instead of 36.8 ([club-3090 #239](https://github.com/noonghunna/club-3090/discussions/239)).

Running llama.cpp is therefore diagnostic. It is a second, independent MTP implementation, and the only working native NVFP4 W4A4 prefill path on sm_120.

## Code acceptance of 2.1 tokens per round points to a fixable drafter defect

### Published runs draft code far better than NInferno does

Google's 31B assistant is a **~0.5B drafter with 4 layers (3 local, 1 global) at width 1024**. It cross-attends to the target's KV cache, uses its own embedder and keeps the full 262K-vocabulary head. The clustered sparse head exists only on E2B/E4B ([Gemma 4 Technical Report](https://arxiv.org/html/2607.02770v1)).

NInferno's 2.6 / 2.1 / 1.6 tokens per round at depth 3 imply per-draft acceptance of roughly 53%, 37% and 20%. Published runs are well above that:

| Source | Setting | Acceptance | Tokens/round |
|---|---|---|---|
| NInferno (current) | Depth 3: math / code / prose | ~53% / 37% / 20% | 2.6 / 2.1 / 1.6 |
| llama.cpp ([PR #23398](https://github.com/ggml-org/llama.cpp/pull/23398)) | Q4 target, n=3 | 0.627 | ≈2.9 |
| ik_llama.cpp ([point.free](https://point.free/blog/gemma-4-mtp/)) | Draft-max 3 | 74% | ≈3.2 |
| Unsloth ([MTP README](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/blob/main/MTP/README.md)) | Q4_K_M target, n=4 | 0.78 | ≈4.1 |

On an H100, MTP's per-category speedups were **coding 3.82x, math 3.37x, writing 1.96x and roleplay 1.56x** ([Jarvislabs](https://jarvislabs.ai/blog/gemma-4-mtp-vs-dflash-benchmark)). Code should be the drafter's strongest suit, not its weakest. Target quantization does not explain the gap: Q4 and Q8 targets gave 0.627 and 0.616 at n=3 in the same llama.cpp test. A llama.cpp tester also saw prefill fall from ~900 to ~700 tok/s with MTP on, although the assistant needs no prefill of its own. Checking that NInferno's 1.4–1.5K prefill is not taxed the same way costs minutes.

### Four suspects sit where NInferno departs from reference code

1. **KV visibility.** The drafter reads the target's cache directly, and every published KV transform has broken it:
   - In llama.cpp, q8_0 KV gave **0% acceptance until a missing Hadamard rotation on Q was added** ([llama.cpp PR #23398](https://github.com/ggml-org/llama.cpp/pull/23398)).
   - In ik_llama.cpp, the Hadamard-KV flags nearly wiped out acceptance ([ik_llama PR #1744](https://github.com/ikawrakow/ik_llama.cpp/pull/1744)).

   NInferno's custom K-only FP8 store for the k_eq_v layers is exactly the kind of transform the drafter's global layer must mirror, including how V is recovered from a post-RoPE K.
2. **QAT pairing.** Google states that a QAT target needs an assistant that is also a QAT checkpoint at the same precision ([Google QAT card](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized)). A QAT-matched `gemma-4-31B-it-qat-q4_0-unquantized-assistant` exists ([Janvitos QAT assistant GGUF](https://huggingface.co/Janvitos/gemma-4-31B-it-qat-assistant-MTP-Q8_0-GGUF)).
3. **The 98,304-row vocabulary shortlist.** If the rows are frequency-ranked, FR-Spec's Llama-3 curve suggests only ~3% loss at 64K rows ([FR-Spec](https://arxiv.org/html/2502.14856v1)). If they are simply the first 98,304 token IDs, every higher-ID token can never be drafted, code symbols and whitespace runs included. The full head costs only ~0.3 ms per draft step (268 MB at q8 over 896 GB/s), against a ~27 ms target step.
4. **Input scaling.** A missing ~sqrt(hidden) embedding scale cut a Gemma DFlash drafter's mean acceptance length from 4.38 to 3.77 ([llama.cpp PR #29802](https://github.com/ggml-org/llama.cpp/pull/29802)). For the 31B the factor is sqrt(5376) ≈ 73.

Temperature is a lesser factor. On Qwen3, acceptance at temperature 1.0 was ~4% (EAGLE-3) to ~15% (DFlash) lower than greedy ([DFlash paper](https://arxiv.org/html/2602.06036)), so greedy numbers must anchor the comparison.

If these fixes bring code and math to the published ~2.9–3.2 tokens/round, decode on those categories rises by roughly the same ratio, 20–40%, because round cost barely changes.

### Depth should adapt to category and context

Depth policy is the next lever. In llama.cpp the best fixed depth was n=3 for a Q4 target and n=4 for Q8, and per-position acceptance decays fast for every method ([vLLM AMD blog](https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus)). At 1.6 tokens/round on prose, most depth-3 drafts are wasted drafter passes. Google's Transformers integration offers a heuristic schedule that adds 2 after a fully accepted draft and subtracts 1 after any rejection ([Google MTP docs](https://ai.google.dev/gemma/docs/mtp/mtp)). Combining that schedule with a confidence stop and per-category starting depths is a small patch. Modified bodies favor shallow drafts: an abliterated 26B-A4B running the vanilla drafter did best at n=1, with acceptance falling from 69% to 39% by n=4 ([ai-muninn](https://ai-muninn.com/en/blog/dgx-spark-huihui-gemma4-fp8-mtp-34pct)).

Long context erodes MTP, and the architecture explains part of the reason. One NVFP4 setup ran 52.5 tok/s with MTP against 22.3 without at a fresh start, but the gain had shrunk to about 1x by 100K tokens ([DEV Community](https://dev.to/gde/the-local-model-that-doesnt-sleep-gemma-4-mtp-as-a-marathon-engine-4c9)). This assumes the drafter's one global layer attends the full context as the target's global layers do. Under that assumption, the config arithmetic says that layer reads a full global-layer slice of the target cache at every draft step. That is about 0.5 GiB at 262K in FP8, comparable to the drafter's own weights, so draft cost roughly doubles at full context. The target verifies every token, so that layer's attention can be windowed or top-k-sparsified with no risk to output quality; only acceptance is at stake.

### DFlash wins code, MTP wins prose

Among alternative drafters, only DFlash earns a place. The z-lab drafter is 913 MB at Q4_K_M ([Anbeeld GGUF](https://huggingface.co/Anbeeld/gemma-4-31B-it-DFlash-GGUF)).

| Setting | Code / math | Prose | Source |
|---|---|---|---|
| Single 3090 | **2.40x** | 1.28x (narrative) | [club-3090 #239](https://github.com/noonghunna/club-3090/discussions/239) |
| 4090, temperature 0.7 | 3.22–3.76x | — | [DataCamp](https://www.datacamp.com/tutorial/how-to-speed-up-local-llms-with-dflash-speculative-decoding) |
| H100, vs MTP | Ties overall: 3.03x vs 3.11x | Loses: 1.53x vs 1.96x | [Jarvislabs](https://jarvislabs.ai/blog/gemma-4-mtp-vs-dflash-benchmark) |

That split argues for a router that sends code, math and thinking to DFlash and prose to MTP.

The other drafters do not pay their way on this card:

- **EAGLE-3:** RedHat's 2B head reaches 3.10–3.15 tokens/round on code and math at k=3 ([RedHatAI EAGLE-3](https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.eagle3)), but lost to MTP head-to-head on AMD.
- **DSpark:** RedHat's 3B drafter posts the best acceptance lengths of any 31B drafter, 6.62 on HumanEval and 4.57 on writing at N=7 ([RedHatAI DSpark](https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.dspark)). It runs only in vLLM, and an independent AMD run did not reproduce its edge ([vLLM AMD blog](https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus)).
- **E2B as a draft model:** it costs 3–5 GB for +29% on a 5090 ([Sleeping Robots](https://sleepingrobots.com/dreams/speculative-decoding-gemma4-strix-halo/)).

Model-free drafting is the cheap complement. SuffixDecoding reached 2.5x mean on SWE-Bench and 1.96–3.12x on code edits, but only 1.0–1.28x on chat ([Snowflake](https://www.snowflake.com/en/blog/engineering/suffixdecoding-arctic-inference-vllm/)). Layering an n-gram or suffix proposer over MTP therefore pays off for agentic loops, RAG quoting and thinking-mode restatement without costing anything on chat.

## Global attention, not GEMMs, sets the speed limit past 180K tokens

### GeForce Blackwell halves FP32 accumulation, not INT8

GeForce Blackwell's tensor rates explain both what is possible and what is not. Per SM per clock, the whitepaper tables work out to:

| Format | FLOP or OP per clock per SM | Laptop peak at 2,160 MHz (82 SMs) |
|---|---|---|
| FP4 (block-scaled) | **4,096** | ~725 TFLOPS |
| INT8 | **2,048** | ~363 TOPS |
| FP8 with FP16 accumulate | **2,048** | ~363 TFLOPS |
| BF16 with FP32 accumulate | **512** | ~91 TFLOPS |

These rates match Ada except for FP4 ([NVIDIA RTX Blackwell whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)). INT8 is not halved; FP32 accumulation is. The stock 175 W part boosts to about 2.0 GHz rather than the 2,160 MHz maximum ([Notebookcheck](https://www.notebookcheck.net/RTX-5090-laptop-GPU-delivers-impressive-performance-uplift-with-250-Watt-TGP.1063500.0.html)). sm_120 also lacks the datacenter features: it has only warp-level `mma.sync`, no tcgen05 or TMEM, and 99 KB of shared memory per SM ([lna-lab SM120 NVFP4](https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm)).

### Short prompts: GEMMs run at llama.cpp-class utilization

Short-context prefill is a GEMM problem with real but expensive headroom. At ~58.6 GFLOP per token, NInferno's 1.45K tok/s is ~85 TFLOPS. That is **about 23–27% of INT8 peak, the same utilization llama.cpp's Q4_K path gets on a desktop 5090** ([Hardware Corner](https://www.hardware-corner.net/rtx-5090-llm-benchmarks/)).

- **INT8 headroom.** Colfax's NVFP4 GEMM reached 83% of FP4 peak on an SM120 card, using TMA loads, warp specialization and a persistent schedule ([Colfax](https://research.colfax-intl.com/optimizing-an-nvfp4-blockscaled-gemm-on-rtx-pro-6000-blackwell-gpu-sm120/)). A CUTLASS-style INT8 kernel at 60% of peak therefore has a theoretical ceiling near 3.3K tok/s. The 175 W limit will pull clocks below that: the desktop 5090 lost 14% of pp512 going from 600 W to 400 W ([llama.cpp discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013)).
- **NVFP4 needs W4A4.** The doubled FP4 rate applies only to block-scaled W4A4, which explains why NInferno's NVFP4 test was slower (~1.27K tok/s). llama.cpp's native W4A4 path lifted Qwen3.6-27B pp512 from 3,295 to 5,547 tok/s ([InsiderLLM](https://insiderllm.com/guides/fp4-inference-llamacpp-nvfp4-mxfp4/)). NVIDIA's own reviewer warned that "quantizing activations to 4-bit undeniably hurts quality" ([llama.cpp PR #22196](https://github.com/ggml-org/llama.cpp/pull/22196)).
- **Marlin is a dead end.** Marlin-style W4A16 is structurally capped at a quarter of the INT8 rate and is not worth porting.

### Ten head_dim-512 layers dominate long prompts

Long-context prefill is an attention problem. Causal attention over the ten global layers costs 327,680·N² FLOPs (derived from the config), which overtakes the GEMMs at about 179K tokens:

| Prompt length | Global-attention FLOPs | Attention ÷ GEMM | Est. laptop prefill time (85 TFLOPS GEMM, 60 TFLOPS attention) |
|---|---|---|---|
| 24K | 198 TFLOP | 14% | ~20 s |
| 128K | 5,630 TFLOP | 73% | ~3 min (≈0.7K tok/s average) |
| 256K | 22,520 TFLOP | 147% | ~9 min (≈0.47K tok/s average) |

No FlashAttention build covers head_dim 512 on sm_120 ([flash-attention #2427](https://github.com/Dao-AILab/flash-attention/issues/2427); [FA4-SM120](https://huggingface.co/SecondNatureComputing/flash-attn-4-sm120)):

- FA2 stops at head_dim 256.
- FA3 runs only on Hopper.
- The community FA4 SM120 port stops at 128, because 99 KB of shared memory cannot hold larger Q tiles.

A custom kernel is unavoidable, and its precision choice dominates:

- **FP16 accumulate or FP8.** Moving QKᵀ and PV from BF16 with FP32 accumulate (512 per clock per SM) to either of these (1,024) **cuts estimated 128K prefill by ~25% and 256K prefill by ~34%**.
- **INT8 QKᵀ.** At 2,048 per clock per SM, it pushes the 256K saving toward half.
- **Quantized attention in general.** SageAttention3 shows how far it can go on this silicon: 1,038 TOPS on a desktop 5090, about 5x FA2 ([SageAttention3](https://arxiv.org/html/2505.11594v3)).
- **K=V tile reuse.** One streamed tile serves both QKᵀ and PV, which halves shared-memory use and bandwidth for the global stream.

The same math reaches decode. By the config arithmetic, verifying 4 tokens at 262K costs ~0.7 TFLOP of global attention. That is ~11 ms at 60 TFLOPS, comparable to the ~8 ms needed to stream the 5.4 GB FP8 cache. At full context, attention precision and KV bytes therefore jointly set MTP round time.

### Decode kernels and the 175 W envelope offer modest gains

Decode kernels have little left to give.

- **Bandwidth.** NInferno's ~640 GB/s matches or beats the 67% llama.cpp reaches on this exact laptop GPU (Llama-2 7B tg128, 158.1 tok/s, [llama.cpp discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013)).
- **Megakernels.** Research megakernels reach 62–78% of H100 bandwidth ([Cohere](https://cohere.com/blog/megakernels)). One hand-written megakernel on an RTX 5080 ran 1.65x slower than vLLM's CUDA-graph path ([worklog](https://emre570.bearblog.dev/megakernel-decode/)).
- **CUDA graphs.** NVIDIA measured at most 1.2x on H100 from CUDA graphs, with smaller gains on larger models ([NVIDIA blog](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs)).

Realistic headroom is 10–20%. Most of it comes from capturing the draft loop in a graph, cutting launches, and re-sweeping route tables that were tuned on the 84-SM desktop 5080. The laptop has 82 SMs and possibly a smaller L2: the 70-SM 5070 Ti ships with 48 MB instead of 64 MB ([whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)).

The hardware levers split cleanly by phase:

- **Decode is memory-bound.** A desktop 5090 ran tg128 at 245.0 tok/s at 400 W and 248.6 at 600 W ([llama.cpp discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013)), so only memory clock moves decode.
  - Raising GDDR7 from 28 to 30–32 Gbps (+7–14%) should lift decode roughly in proportion if stable. The offset is set through `nvmlDeviceSetClockOffsets` ([NVML docs](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceCommands.html)).
  - One laptop rejected P-state-0 offsets ([NVIDIA forum](https://forums.developer.nvidia.com/t/nvmldevicegetminmaxclockofpstate-nvmldevicesetclockoffsets-issues/318332)), and an unstable GDDR7 overclock can surface as silent errors.
- **Prefill is power-bound.** A 250 W shunt mod lifted clocks from ~2.0 to ~2.7 GHz ([Notebookcheck](https://www.notebookcheck.net/RTX-5090-laptop-GPU-delivers-impressive-performance-uplift-with-250-Watt-TGP.1063500.0.html)).
  - On Linux, the 175 W Dynamic Boost tier depends on `nvidia-powerd`. An RTX 5070 Laptop sat at 50 W until it was enabled ([Arch forum](https://bbs.archlinux.org/viewtopic.php?id=313511)).
  - The only undervolt available is a capped maximum clock plus a positive V/F offset.
- **All tuning belongs on the Proxmox host.** These changes must run as root on the host, not in the LXC, and `nvidia-smi -lgc/-lmc` is unreliable on GeForce Blackwell ([NVIDIA forum](https://forums.developer.nvidia.com/t/how-to-force-lock-sm-and-memory-clocks-on-rtx-5090-headless-linux/348794)).

## A 4-bit rotated K-only cache is the one safe memory win left

At 262K tokens the memory budget is tight but not exhausted, and only the global cache is worth compressing further. The figures below are derived from the config and published file sizes:

| Component at 262,144 tokens | Today: FP8 K-only | 4.5-bit rotated K-only |
|---|---|---|
| Unsloth QAT UD-Q4_K_XL weights | 16.10 GiB | 16.10 GiB |
| MTP assistant (Q8_0 GGUF, 515 MB) | 0.48 GiB | 0.48 GiB |
| Global K(=V) cache | 5.04 GiB | 2.81 GiB |
| SWA cache, FP8, 1,024-token window | 0.39 GiB | 0.39 GiB |
| Subtotal before activations and CUDA context | 22.0 GiB | 19.8 GiB |
| Room for a DFlash Q4_K_M drafter (0.85 GiB) at 262K | Only by giving up ~43K tokens | Yes |

The case for 4-bit rests on the only 256K-scale evidence available. In vLLM's TurboQuant study on Qwen3-30B, MRCR AUC up to 256K was **45.8 at BF16, 43.1 at FP8 and 42.3 at 4-bit, but 31.2–33.5 at 3-bit**. The 3-bit losses were concentrated in the 128K–256K band, and 3-bit also dropped about 20 points on reasoning ([vLLM TurboQuant blog](https://vllm.ai/blog/2026-05-11-turboquant)). On Gemma 4 31B itself, ik_llama.cpp measured perplexity of 5.2306 with FP16 KV, 5.2311 with q8_0 and 5.2323 with q6_0 plus Hadamard ([ik_llama.cpp #1607](https://github.com/ikawrakow/ik_llama.cpp/issues/1607)).

Three warnings temper this:

- **Long reasoning.** Rotated q4_0 KV in llama.cpp still dropped gpt-oss-20b AIME25 from 37.9% to 21.7% ([llama.cpp PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038)), so long reasoning needs its own gate.
- **Disputed sensitivity.** A localbench KLD test called Gemma 4 unusually sensitive even at q8_0 (KL 0.108). That result conflicts with the perplexity data above and is disputed ([localbench](https://localbench.substack.com/p/kv-cache-quantization-benchmark)).
- **Dequant cost.** Bandwidth arithmetic says the 4-bit store cuts per-token traffic at 262K from ~23.1 GB to ~20.7 GB, worth roughly +12% decode. But q4 KV cost 13–29% of decode speed at 16K depth in llama.cpp on a desktop 5090 ([witcheer benchmarks](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/README.md)). The speed gain only appears if dequantization is fused into the attention kernel.

K=V reshapes the design:

- **Key-grade precision.** One tensor serves as both key and value, so the usual asymmetric "K8V4" split does not apply. Keys carry channel outliers and need more bits than values ([Quantize What Counts](https://arxiv.org/abs/2502.15075); [KVQuant](https://arxiv.org/abs/2401.18079)), which sets a 4-bit floor with rotation for the stored tensor.
- **Static dimensions.** p-RoPE rotates only 128 of the 512 dims. The remaining 384 static dims suit per-channel scales plus block-Hadamard, which gives KVQuant's pre-RoPE recipe for free.
- **NVFP4 as the container.** On Blackwell, NVFP4 stores E2M1 values with an E4M3 scale per 16 elements. NVIDIA reports under 1% accuracy loss through RULER 64K ([NVIDIA blog](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)).
- **Leave SWA alone.** The SWA cache should stay at FP8 or BF16: compressing it saves at most 0.2–0.4 GiB, and it is the suspected source of Gemma's KV sensitivity.

Everything more aggressive should wait:

- **Token eviction.** Gemma 4 already behaves like a trained DuoAttention split, with only 40 of 840 KV heads global. Evicting tokens attacks the model's only long-range path. In a query-agnostic audit, SnapKV scored below a trivial "keep start plus recent" baseline, and only KeyDiff consistently beat that baseline ([arXiv 2607.11942](https://arxiv.org/abs/2607.11942)).
- **Host offload.** Offload with sparse retrieval (ShadowKV, RetrievalAttention) remains research code for Llama-class models ([ShadowKV](https://arxiv.org/html/2410.21465v1)).

A more practical gap is multi-turn prompt reuse. llama.cpp users report full prompt re-evaluation on every Gemma 4 request ([llama.cpp #21468](https://github.com/ggml-org/llama.cpp/issues/21468)). Gemma's chat template strips prior thoughts from model turns ([Google model card](https://huggingface.co/google/gemma-4-31B-it)), which forces the prompt to diverge mid-history. If NInferno re-prefills on turn two, saving SWA checkpoints at turn boundaries would avoid minutes of 128K+ prefill per turn. That is a bigger effective-prefill gain than any GEMM rewrite.

## QAT UD-Q4_K_XL already sits on the 16 GiB quality frontier

Measured against the unquantized QAT checkpoint, the two grid-faithful Q4 files are about 6x closer than everything else ([wazimondo Q4_0_V2](https://huggingface.co/wazimondo/gemma-4-31B-it-qat-Q4_0_V2)):

| File | Mean KLD | 99.9% KLD | Same top-p |
|---|---|---|---|
| pajari Q4_0_V2 | 0.00199 | 0.229 | 99.42% |
| Unsloth QAT UD-Q4_K_XL (16.10 GiB) | 0.00220 | 0.258 | 99.34% |
| Google qat-q4_0-gguf (16.44 GiB) | 0.01325 | 1.077 | 97.67% |
| Stock llama.cpp Q4_0 | 0.01396 | 1.140 | 97.56% |

Perplexity ranked these files backwards, so KLD, especially the 99.9th percentile, is the metric to gate on. Unsloth traces the gap to conversion: a naive QAT-to-Q4_0 conversion matches the QAT weights byte-for-byte only 24.77% of the time, against 99.96% for its method ([Unsloth QAT docs](https://unsloth.ai/docs/models/gemma-4/qat)). Google's own GGUF is strictly dominated.

pajari's Q4_0_V2 is the only file worth an A/B test. Its KLD is equal or better, and its plain Q4_0 blocks decode faster than K-quants on Gemma 4 ([3h4x](https://3h4x.github.io/tech/2026/06/08/gemma4-qat-vs-non-qat)). Its size is unstated, so it must be checked for fit.

The open question is whether QAT beats a good post-training quant measured against the *original* BF16 model. Nobody has measured that for the 31B. A 12B study found that QAT gave no measurable gain on routing tasks ([Google AI forum](https://discuss.ai.google.dev/t/gemma-4-qat-q4-0-vs-ptq-q4-k-m-on-an-rtx-5070-ti-opposite-sign-routing-effects-across-released-pairs-and-68-77-greedy-output-divergence-sequel/177799)).

Other formats either cost bytes or lack evidence:

- **EXL3.** On a Gemma 4 12B proxy, EXL3 at 4.0 bpw beat UD-Q4_K_XL on KLD (0.0248 vs 0.0361), with 2.2x faster prefill but 15–21% slower decode ([Atomic Chat EXL3 vs GGUF](https://atomic.chat/blog/guides/exl3-vs-gguf)). Its high-precision embeddings push file size toward Q5_K_M. It was measured against non-QAT references, it needs trellis kernels NInferno lacks, and ExLlamaV3 has no MTP.
- **NVFP4 checkpoints.** They recover about 99% of BF16 on MMLU-Pro and GPQA ([RedHatAI NVFP4](https://huggingface.co/RedHatAI/gemma-4-31B-it-NVFP4)) but weigh at least 18.5 GB. A vLLM maintainer noted that the MTP assistants were trained against the base models, not NVFP4 variants ([vLLM PR #41745](https://github.com/vllm-project/vllm/pull/41745)).
- **Requantized QAT weights.** Converting QAT weights to any grid other than Q4_0 forfeits their near-zero error by construction.
- **Higher-bit weights.** Spending freed KV memory on higher-bit weights buys nothing. Unsloth ships one file per QAT model because higher precisions move away from the QAT reference ([Unsloth QAT docs](https://unsloth.ai/docs/models/gemma-4/qat)).

If an uncensored body is required, the GGUF conversion matters more than the abliteration method:

- **Heretic ARA.** Heretic's Arbitrary-Rank Ablation posts the lowest KL at low refusal rates: **0.012 with 5/100 refusals** for trohrbaugh's build ([trohrbaugh](https://huggingface.co/trohrbaugh/gemma-4-31b-it-heretic-ara)).
- **llmfan46's QAT-based Heretic.** It is the only variant with a full-MMLU check: 84.46%, against 86.17% for the QAT original ([llmfan46 QAT heretic](https://huggingface.co/llmfan46/gemma-4-31B-it-qat-q4_0-unquantized-uncensored-heretic)).
- **huihui.** huihui's classic abliteration publishes no metrics, and it scored ~3x higher KL than Heretic on Gemma 3 12B ([p-e-w/heretic](https://github.com/p-e-w/heretic)).
- **Naive conversion.** A naive llama-quantize Q4_0 adds mean KLD of ~0.093 by Unsloth's measure, which is more damage than the abliteration itself.

The best uncensored route is therefore llmfan46's QAT Heretic converted with a grid-faithful Q4_0 encoder, with MTP depth re-swept because modified bodies favor shallower drafts.

Dropping to the 26B-A4B model triples decode speed, but costs 22 points on MRCR 128K (44.1 vs 66.4) and 432 Codeforces Elo ([Google model card](https://huggingface.co/google/gemma-4-31B-it)). That is the wrong trade for a long-context workload.

## A ranked test plan front-loads diagnostics and free switches

The plan orders work by expected gain per unit of effort. Every speed change passes the same gates:

- Per-category tokens/round and tok/s at 4K, 32K, 128K and 262K.
- Prefill at 2K, 32K, 128K and 256K.
- Mean and 99.9% KLD against the QAT BF16 reference.
- MRCR v2 8-needle at 128K, compared against Google's published 66.4%.

Steps 1–6 need no new subsystems and should run in the first few days. Steps 7–15 are code projects or engine trials, ordered so that earlier steps enable later ones; for example, the 4-bit cache frees the memory the DFlash drafter needs at 262K.

| Rank | Experiment | Expected gain | Effort and new code | Go/no-go gate |
|---|---|---|---|---|
| 1 | Baseline harness plus hardware audit: true DRAM ceiling (nvbandwidth), real L2 size, sustained clocks and power during prefill, `nvidia-powerd` running on the Proxmox host, prefill with drafter on vs off, turn-two re-prefill check | Enables every later step; large if Dynamic Boost is off | Hours; no code | 175 W reached; prefill unchanged with drafter on |
| 2 | MTP acceptance A/B: QAT-matched assistant; full 262,144-row head vs 98,304 shortlist; reference BF16 K+V vs FP8 K-only store as the drafter sees it; embedding and hidden-state scaling; greedy vs temperature 1.0 | +20–40% decode on code and math if acceptance reaches the published 2.9–3.2 tokens/round | Config plus small patches | Code ≥2.8 tokens/round at depth 3, greedy |
| 3 | Mainline llama.cpp reference on the laptop: build with `-DGGML_CUDA_FA_ALL_QUANTS=ON` and arch 120, run `--spec-type draft-mtp --spec-draft-n-max 3` on the same GGUF, plus an NVFP4 GGUF for prefill | Diagnostic: independent acceptance and NVFP4 prefill numbers on this exact GPU | Half a day; external engine | Explains any acceptance gap; sizes the W4A4 gain |
| 4 | Adaptive depth: +2/−1 schedule, confidence stop, per-category starting depth, shallower drafts at long context | Estimated single-digit to low double-digit %, mostly on prose and long context | Small patch | Beats the best fixed depth in every category |
| 5 | Memory-clock offset and V/F undervolt on the Proxmox host | Decode rises roughly in line with a +7–14% memory clock; prefill gain unmeasured | No code; stability risk | Greedy outputs and KLD unchanged over a multi-hour soak |
| 6 | pajari Q4_0_V2 vs UD-Q4_K_XL | Equal or better KLD; possibly faster Q4_0 decode | Config if NInferno has a Q4_0 path; check fit | Lower 99.9% KLD and still fits 262K |
| 7 | n-gram/suffix proposer layered over MTP | 2–5x on edits, RAG and agent loops; ~1x on chat | Small to medium new code | No regression on chat |
| 8 | 4.5-bit Hadamard-rotated K-only global cache (NVFP4 or q4_0/iq4_nl, per-channel scales on the 384 static dims) | Frees ~2.2 GiB at 262K; ~+12% decode at 262K, but only with fused dequant | Medium new code | MRCR-128K within ~1 point of FP8; long-reasoning check passes |
| 9 | Low-precision global attention for prefill and verify: FP16 accumulate or INT8 QK with FP8 PV, K=V tile reuse, 8-head GQA packing | At 2x attention rate, ~25% faster 128K and ~34% faster 256K prefill; faster long-context verify | New kernel; medium-high effort | KLD and MRCR unchanged |
| 10 | DFlash: pre-test with beellama.cpp or llama.cpp `draft-dflash`, checking the ~73x embedding scale; if it wins, build a NInferno module with a code/math vs prose router | 2.4–3.8x over no speculation on code, vs ~1.5–2x for MTP today | Half-day pre-test; high-effort integration | Pre-test beats tuned MTP on the user's own code prompts |
| 11 | Windowed or top-k attention for the drafter's global layer beyond ~64K tokens | Restores MTP gains at 100K+; size of gain unknown | Medium; no risk to output quality | Acceptance loss smaller than draft-time saving |
| 12 | SWA checkpoints for multi-turn prefix reuse, only if step 1 shows re-prefill | Avoids minutes of re-prefill per turn at 128K+ | Medium | Turn-two time-to-first-token scales with new tokens only |
| 13 | CUTLASS-style INT8 GEMM rewrite: TMA loads, warp specialization, persistent schedule, re-swept for 82 SMs and 99 KB shared memory | Ceiling of ~2x short-context prefill; less at 175 W | High | ≥1.3x prefill at 4K |
| 14 | NVFP4 W4A4 prefill path | +58–68% prefill in desktop llama.cpp | High; at least 1 GiB more weights; quality cost from 4-bit activations | Only if step 3 shows the gain on the laptop and KLD holds |
| 15 | EXL3 4.0 bpw KLD check in TabbyAPI | Quality per byte, measured against the original BF16 model | Low to test; very high to integrate | Must beat QAT UD-Q4_K_XL against the same reference |

Several ideas drop off the list on current evidence:

- **vLLM, SGLang and TensorRT-LLM:** no 24 GB single-GPU path at long context.
- **EAGLE-3:** MTP-level acceptance at four times the size.
- **E2B as a draft model:** costs 3–5 GB.
- **DSpark:** watch for a GGUF path.
- **3-bit KV and global-layer eviction:** their damage concentrates at 128–256K.
- **Megakernels and L2 persistence:** high effort, little bandwidth headroom left.
- **Google's QAT Q4_0 GGUF:** dominated by the Unsloth and pajari files.

## Conclusion

The weights are no longer the bottleneck for speed or quality. Decode already runs near the bandwidth ceiling with a frontier-quality 4-bit file. The remaining multipliers live in two places: the speculative loop's correctness, and the ten global layers into which Gemma 4 concentrates its long-range work. NInferno's existing K=V advantage is the template for the next round. The same "one tensor, two roles" insight applies to three places: tile reuse in the attention kernel, the drafter's view of the cache, and the quantizer's key-grade 4-bit floor. No general engine has yet done this kind of architecture-specific engineering for this model.

The second implication is methodological. Every lever with real upside (4-bit KV, quantized attention, W4A4 activations, drafter approximations) trades bits for speed where Gemma 4's published evidence is thin or disputed. A fixed KLD-plus-MRCR gate run on every change is worth more than any single optimization. If steps 1–4 land, code and math decode should move toward the upper half of the 58–80 tok/s range that published MTP results project for this bandwidth. Going beyond that, or cutting 262K prefill from roughly nine minutes to a few, depends on the attention kernel, not on the next engine release.
