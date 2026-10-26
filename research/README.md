# Research and reports

| Date | Report | What it covers |
|---|---|---|
| 2026-10-25 | [Engines and options](2026-10-25-research-engines-and-options.md) | the research before the build: engines (llama.cpp, ik_llama.cpp, ExLlamaV3, vLLM, SGLang, TensorRT-LLM) on 24 GB, quantization formats, speculative decoding, KV compression, prefill on Blackwell consumer, community numbers |
| 2026-10-25 | [Day 1 results](2026-10-25-day1-results.md) | the build: QAT drafter, K-only KV, NVFP4 W4A4 with the GeGLU split, INT8 and then FP8 attention, the quality isolation of NVFP4, needles, the two deployed profiles |
| 2026-10-25 | [QAT Q4 fast prefill design](2026-10-25-qat-q4-fast-prefill-design.md) | why QAT Q4 prefills at about 1.5K on this card (ggml MMQ at about 0.33 of INT8 peak), what NInferno already has, and a plan for an INT8 Q4_0 kernel at 2.0–2.6K |
| 2026-10-26 | [Night 2: context, quality, vision, speed](2026-10-26-night2.md) | 262K on the fast profile at production speed, NVFP4 builds with higher quality (LilaRest turbo with Q/K/V packed: MMLU-Pro 178 at production prefill), vision with MTP after an engine fix, agent-turn speed, uncensored builds |
| 2026-10-26 | [Night 1: decode tuning](2026-10-26-night1-decode.md) | 7 MTP drafts, the QKV and Q6-head schedules (+8–18% decode), what did not pay, agent-turn latency, tool calling, the max-context profile at 231K |
