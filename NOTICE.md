# Notice

## This repository

Our scripts, configs and documents are under the MIT licence ([`LICENSE`](LICENSE)), Copyright 2026 Leon Bollerup.

## Patches

- **[`patches/ninferno/`](patches/ninferno/)** modify [NInferno](https://github.com/Brontir/NInferno) and are under the Apache License, Version 2.0, like NInferno.
  - NInferno is derived from [ninfer](https://github.com/Neroued/ninfer), Copyright 2026 Neroued and contributors, Apache-2.0.
  - It takes kernels from [ninfer-all](https://github.com/iamwavecut/ninfer-all) (Valeriy Selitskiy and contributors, Apache-2.0) and [ninfer-v100-sm70](https://github.com/Flo5k5/ninfer-v100-sm70) (Flo5k5, Apache-2.0). NInferno's own NOTICE lists those files.
  - Our patches change NInferno's Gemma 4 attention kernel, its Gemma 4 model code, two NVFP4 shape schedules and one Q6 shape route, and one admission entitlement count in its program planner.
- **[`config/templates/gemma4-agent.jinja`](config/templates/gemma4-agent.jinja)** is Google's Gemma 4 chat template ([`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it), Apache-2.0) with our changes, under Apache-2.0.
- **[`patches/llama.cpp/`](patches/llama.cpp/)** modifies [llama.cpp](https://github.com/ggml-org/llama.cpp) (the ggml authors, MIT) and is under MIT, like llama.cpp.
- **[`patches/experiments/`](patches/experiments/)** are diffs against NInferno, under Apache-2.0.

## Not included, referenced

| Component | Source | Licence |
|---|---|---|
| Gemma 4 31B and its MTP assistant | Google, e.g. [`google/gemma-4-31B-it-qat-q4_0-unquantized-assistant`](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized-assistant) | [Gemma 4 licence](https://ai.google.dev/gemma/docs/gemma_4_license) (Apache-2.0) |
| QAT NVFP4 conversion | [`melcheikh/gemma-4-31B-it-qat-NVFP4-Blackwell`](https://huggingface.co/melcheikh/gemma-4-31B-it-qat-NVFP4-Blackwell), made with NVIDIA Model Optimizer | Apache-2.0, Gemma 4 licence |
| QAT UD GGUF | [`unsloth/gemma-4-31B-it-qat-GGUF`](https://huggingface.co/unsloth/gemma-4-31B-it-qat-GGUF) | Apache-2.0, Gemma 4 licence |
| Other checkpoints compared | Pajari (Q4_0_V2), llmfan46 (Heretic), huihui-ai (abliterated), Anbeeld (DFlash) | their own licences, on their model cards |
| NInfer web UI | iNordic (Strata-derived) | not included; only our service files are |
| Benchmarks | GSM8K (OpenAI), MMLU (Hendrycks et al., `cais/mmlu`), MMLU-Pro (TIGER-Lab), HumanEval (OpenAI) | MIT each; rebuilt locally by `bench/data/make_subsets.py`. [`results/quality/`](results/quality/) holds only item scores, gold answers and predictions |
| Needle haystack | NInferno's `examples/cli/messages/long_niah_256k.json` (Federal Register text) | read from the NInferno tree, not copied |

No model weights, firmware or datasets are included.
