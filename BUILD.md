# Build: from a container with an RTX 5090 Laptop GPU to both profiles serving

This is the route we took. The paths are ours:
- **Engine:** two trees. `/opt/NInferno-up` (upstream `8359583` + patches 01–16) serves the fast unit from `build/apps-up1`; `/opt/NInferno` (`9895ebc` + patches) serves the max-context unit from `build/apps-final7` (patches 01–09). Production runs these copies of the binaries; later builds in the trees don't touch them.
- **Converter:** `/opt/llama.cpp`, the patched one, used only to convert the drafter.
- **Models:** `/opt/models/dl` and `/opt/models/dl2`.
- **Benchmarks:** `/root/gbench`, holding the contents of [`bench/`](bench/).

Adjust them to your system. Each step says how to check it before you go on.

## 0. Hardware and container

| Item | Ours | Notes |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU (GB203, sm_120, 82 SMs, 24 GB GDDR7, 896 GB/s) | driver 595.58.03 on the host, VBIOS 98.03.95.00.07 |
| Host | Proxmox VE, kernel 6.17.13-2-pve | the same GPU is also passed to a second container; only one of them may serve at a time, so stop the other before you start here |
| Container | LXC, Ubuntu 24.04, 8 cores, 48 GB RAM | `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm*` and `/dev/nvidia-caps` passed in, with the user-space driver matching the host |
| Toolchain | CUDA 13.2 at `/usr/local/cuda-13.2`, CMake 3.28, Ninja 1.11, GCC 13.3, Python 3.12 | |
| Disk | about 60 GB for the models below, plus the downloads while converting | |

**Check:** `nvidia-smi` in the container shows the RTX 5090 Laptop GPU with 24 GB and nothing else running on it.

## 1. NInferno with our patches

The fast unit runs our sixteen patches on upstream NInferno `8359583`. They were written on the older base `9895ebc`: `git am -3` merges them across, and patch 03 needs its rebased version from [`patches/ninferno/on-8359583/`](patches/ninferno/on-8359583/).

```bash
git clone https://github.com/Brontir/NInferno.git /opt/NInferno-up
cd /opt/NInferno-up
git checkout -b kvshare-up 83595831b790ca2ec9ed6567879063acec2a6c83
P=/path/to/this/repo/patches/ninferno                     # patches 01-16, see patches/README.md
git am -3 $P/000[12]-*.patch $P/on-8359583/0003-*.patch $P/000[4-9]-*.patch $P/001[0-6]-*.patch
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
PATH=/usr/local/cuda-13.2/bin:$PATH ninja -C build ninferno-serve ninferno-gemma4 ninferno ninferno-convert
mkdir -p build/apps-up1 && cp build/apps/ninferno-serve build/apps/ninferno-gemma4 build/apps/ninferno build/apps-up1/
```

In a clone that holds only upstream's history, this series gives exactly the tree of our `kvshare-up` (`afd15fa`).

The max-context unit still runs the build at patch 09 on the original base (`/opt/NInferno/build/apps-final7`); it hasn't been re-measured on newer builds. Step 3 converts with that tree's `build/apps/ninferno-convert` at the branch tip (patch 13 packs the fast model's Q/K/V). To set it up:
1. Clone into `/opt/NInferno` and run `git checkout -b kvshare-main 9895ebc62e2bdfb367083a72cb2c4a714cf41542`.
1. Run `git am $P/00*.patch`; all sixteen apply cleanly on this base.
1. For the max-context binaries, run `git checkout HEAD~7` (patch 09), build as above, and copy them to `build/apps-final7`.
1. Run `git checkout kvshare-main` and build again for the converter.

The full build takes a long time; the kernels are many. Add `-DNINFERNO_BUILD_BENCHMARKS=ON` and `ninja -C build bench/ninferno_linear_bench` if you want the kernel microbenchmark used in [`bench/lb.sh`](bench/lb.sh).

**Check:**
- `git log --oneline -1` shows `serve: opt-in automatic shared-prefix candidates after the tools, and as requested evidence`, our patch 16.
- `build/apps-up1/ninferno-serve --version` prints `NInferno 0.1.0 (built for sm_120, CUDA 13.2)` and lists the GPU.

## 2. llama.cpp with the QAT-aware Q4_0 encoder (for the drafter)

The drafter is converted through GGUF. A plain Q4_0 conversion moves the QAT weights off their int4 grid; our patch recovers each block's BF16 scale and stores the grid exactly ([`patches/llama.cpp/`](patches/llama.cpp/)).

```bash
git clone https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp
cd /opt/llama.cpp
git checkout -b qat-aware-q4_0 2280f0f
git am /path/to/this/repo/patches/llama.cpp/*.patch
python3 -m venv .venv && .venv/bin/pip install -e gguf-py numpy safetensors sentencepiece protobuf transformers torch --extra-index-url https://download.pytorch.org/whl/cpu
```

Our venv has gguf 0.19.0, numpy 2.2.6, safetensors 0.8.0, sentencepiece 0.2.2, transformers 5.20.2 and torch 2.11.0+cpu. Nothing is compiled; the converter is Python.

**Check:** `git log --oneline -1` shows `QAT-aware Q4_0: recover each block's BF16 QAT grid and store it exactly`.

## 3. Models

The download scripts we used are in [`bench/models/`](bench/models/). They use plain `curl` against huggingface.co, and resume.

### 3a. The drafters

The max-context profile drafts with Google's QAT assistant:

```bash
D=/opt/models/dl2
mkdir -p $D/qat-assistant && cd $D/qat-assistant
for f in config.json generation_config.json tokenizer_config.json model.safetensors tokenizer.json; do
  curl -sL -C - --fail --retry 10 -o $f https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized-assistant/resolve/main/$f
done
cd /opt/llama.cpp && GGUF_Q4_0_QAT=1 .venv/bin/python convert_hf_to_gguf.py $D/qat-assistant \
    --outtype q4_0 --outfile $D/qat-assistant-Q4_0-qat.gguf
/opt/NInferno/build/apps/ninferno-convert --model $D/qat-assistant-Q4_0-qat.gguf \
    --recipe gemma4_assistant_published --out $D/qat-assistant.q40qat.ninferno
```

**Check:**
- The GGUF is 279,955,680 bytes, sha256 `349ccc8eea57e012…`.
- The artifact is 277,767,172 bytes. Its `.conversion.json` lists `ggml_q4_0: 19, bf16: 21, fp32: 4`.

The fast profile drafts with Google's regular assistant, which matches its BF16-derived target:

```bash
mkdir -p $D/assistant && cd $D/assistant
for f in config.json generation_config.json tokenizer_config.json model.safetensors tokenizer.json chat_template.jinja; do
  curl -sL -C - --fail --retry 10 -o $f https://huggingface.co/google/gemma-4-31B-it-assistant/resolve/main/$f
done
/opt/NInferno/build/apps/ninferno-convert --model $D/assistant --recipe gemma4_assistant_published --out $D/assistant.q8full.ninferno
```

**Check:** the artifact is 531,091,716 bytes. Its `.conversion.json` lists `q8_g32_fp16: 19, bf16: 21, fp32: 4`.

### 3b. The fast profile's target: LilaRest NVFP4-turbo, Q/K/V packed, with vision

The weights come from LilaRest's NVFP4 of Google's BF16 model. It is text-only, so the vision tower comes from Google's original model, by way of Unsloth's projector file. The QAT repo's projector file is different. [`bench/runs/night2/prep_turbo_vision.sh`](bench/runs/night2/prep_turbo_vision.sh) downloads both.

```bash
D=/opt/models/dl2
# LilaRest/gemma-4-31B-it-NVFP4-turbo, every file, into $D/turbo-src; and the projector:
curl -sL -C - --fail --retry 10 -o $D/it-mmproj-BF16.gguf \
    https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/resolve/main/mmproj-BF16.gguf
cd $D && NINFERNO_CONVERT_NORMED_QKV_DIVISOR=1 /opt/NInferno/build/apps/ninferno-convert --model $D/turbo-src \
    --recipe gemma4_31b_nvfp4 --source vision=$D/it-mmproj-BF16.gguf --components text,vision \
    --out $D/turbo-qkv-vision-nvfp4.r31.ninferno
```

`NINFERNO_CONVERT_NORMED_QKV_DIVISOR=1` (patch 13) packs each layer's query, key and value into one NVFP4 plane, though turbo gives them different global scales. Without it they stay three GEMMs, and prefill is 6–9% slower.

**Check:**
- The artifact is 20,259,160,836 bytes.
- Its `.conversion.json` lists `nvfp4: 240, bf16: 636, q8_g32_fp16: 1, q6_g64_fp16: 1, fp32: 60` and the components `text` and `vision`.
- Every object holding `attention/query` also holds `attention/key`.

### 3c. The max-context profile's target: QAT-UD Q4

```bash
cd /opt/models/dl && curl -sL -C - --fail --retry 10 -o gemma-4-31B-it-qat-UD-Q4_K_XL.gguf \
    https://huggingface.co/unsloth/gemma-4-31B-it-qat-GGUF/resolve/main/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf
/opt/NInferno/build/apps/ninferno-convert --model /opt/models/dl/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf
```

Without `--out`, the converter writes to `.ninferno-cache/gemma-4-31B-it-qat-UD-Q4_K_XL.gemma4_published.ninferno` next to the GGUF. Despite the name, every matrix in this GGUF is Q4_0 on the QAT grid.

**Check:** the artifact is about 17.3 GB, and its `.conversion.json` lists `ggml_q4_0: 241`.

### 3d. Optional: uncensored (Heretic, QAT-aware Q4_0)

[`bench/models/heretic.sh`](bench/models/heretic.sh) downloads `llmfan46/gemma-4-31B-it-qat-q4_0-unquantized-uncensored-heretic` and converts it with the same QAT-aware encoder. Then convert the GGUF as in 3c. It runs on the max-context profile's config at about 108 tok/s on chat, with GSM8K 196, MMLU-Pro 164, MMLU 246 and HumanEval 160: the same as QAT Q4 ([night 2](research/2026-10-26-night2.md#5-uncensored-builds-batches-c-and-i)).

### 3e. The previous fast target: QAT-NVFP4 (for comparison)

```bash
sh bench/models/nvfp4.sh        # melcheikh/gemma-4-31B-it-qat-NVFP4-Blackwell into /opt/models/dl2/qat-nvfp4
cd /opt/models/dl2 && /opt/NInferno/build/apps/ninferno-convert --model /opt/models/dl2/qat-nvfp4 \
    --recipe gemma4_31b_nvfp4 --components text --out /opt/models/dl2/qat-nvfp4.r31.ninferno
```

The `gemma4_31b_nvfp4` recipe keeps the checkpoint's NVFP4 projections, stores the embedding table in Q8 and the head in Q6, and drops the vision tower (`--components text`). The published recipe keeps the vocabulary in BF16, which costs about 2.8 GiB.

**Check:**
- The artifact is 19,107,566,596 bytes.
- Its `.conversion.json` lists `nvfp4: 240, q8_g32_fp16: 1, q6_g64_fp16: 1`.
- The server reports 18.0 GiB of weights at start.

Don't use melcheikh's "-mse-Blackwell" repo instead: its language-model tensors are byte-identical to this one.

## 4. Serve

1. Copy the chat template: `mkdir -p /opt/models/templates && cp config/templates/gemma4-agent.jinja /opt/models/templates/`. It is Google's Gemma 4 template of 2026-07-09 with an agent patch ([`config/README.md`](config/README.md#templates)); both units pass it with `--chat-template`.
1. Copy [`config/systemd/gemma4-31b-fast.service`](config/systemd/gemma4-31b-fast.service) and [`config/systemd/gemma4-31b-262k.service`](config/systemd/gemma4-31b-262k.service) to `/etc/systemd/system/`. Set the paths if yours differ.
1. Run `systemctl daemon-reload && systemctl start gemma4-31b-fast`.

The two units conflict, so starting one stops the other. Enable the one that should come up after a reboot; we enable the fast one: `systemctl enable gemma4-31b-fast`. If the GPU is busy at boot (another container on it), the unit fails and retries every 10 s until it starts.

The first start takes about 25 s: weights, plan and CUDA graphs. The log line `capacity | KV 202,112 tokens, fp8, sliding rings fp8` confirms the context.

**Check**, in order:
1. `curl -s localhost:8080/v1/models` shows `max_model_len 202112`, `"capabilities": {"images": true}` and `"config": "mtp K=7"`.
2. [`bench/verify.py`](bench/verify.py) passes: models, chat, streaming, and three agent-style turns with prefix reuse. The second and third turns should report `cache` around 17,600 and take about 0.5 s.
3. [`bench/gbench.py`](bench/gbench.py) `--pp 2048,8192,32768,65536` lands near the numbers in [`results/results.jsonl`](results/results.jsonl), label `K1 turbo-qkv-vision ctx202112`.
4. [`bench/visiontest.py`](bench/visiontest.py) scores 7/7, and [`bench/needle.py`](bench/needle.py) `--lengths 140000,181900 --depths 0.5` scores 2/2.
5. [`bench/agenttest.py`](bench/agenttest.py) `--label check --runs 3` ends with `"markup_as_text": 0` and `"errors": 0`, and [`bench/textcalltest.py`](bench/textcalltest.py) prints `TEXTCALL leaks 0`.

[`bench/deploy_fast.sh`](bench/deploy_fast.sh) stops anything on 8080 and brings the fast unit up again. Our overnight runners call it after every test batch.

## 5. Optional: the NInfer web UI

The UI is iNordic's NInfer web UI, a Node app: release 0.1.0, commit `1492c70`. It isn't in this repository. Its Monitor reads the engine's `/metrics` (patch 08's counters), so it shows every client of 8080. Put the app in `/opt/ninfer-ui/app` and its bundled Node 22 runtime in `/opt/ninfer-ui/runtime`, then:

```bash
mkdir -p /etc/ninfer-ui/tls && chmod 700 /etc/ninfer-ui/tls
openssl req -x509 -newkey rsa:2048 -nodes -days 825 -subj "/CN=<host-ip>" -addext "subjectAltName=IP:<host-ip>" \
    -keyout /etc/ninfer-ui/tls/key.pem -out /etc/ninfer-ui/tls/cert.pem
chmod 600 /etc/ninfer-ui/tls/*.pem
cp config/ninfer-ui/ninfer-ui.service /etc/systemd/system/
cp -r config/ninfer-ui/ninfer-ui.service.d /etc/systemd/system/   # set NINFER_UI_HOST to your IP in accounts.conf
systemctl daemon-reload && systemctl enable --now ninfer-ui
```

Stats are on `http://<host-ip>:8081`, and sign-in is on `https://<host-ip>:8443`. The first start creates `admin` with a one-time password in `/var/lib/ninfer-ui/initial-admin-password`, which must be replaced at first sign-in.

**Tools.** The UI's tool calling starts switched off. Turn it on as admin under Settings → Tools, or from the shell with [`config/ninfer-ui/tools-on.mjs`](config/ninfer-ui/tools-on.mjs); no restart is needed, and a reload of the page shows "Use tools" in the Sampling drawer. The built-in `current_time` and `calculator` are then on; an MCP server's tools start off and are switched on one by one. Clients that call 8080 directly (agents) need nothing: the engine accepts OpenAI `tools` and returns `tool_calls`, streamed or not.

**Check:** `curl http://<host-ip>:8081/health` shows `"model":"gemma4-31b-fast","max_context":202112,"images":true`.

## 6. Measure

| What | How |
|---|---|
| Prefill and decode | `python3 bench/gbench.py --label NAME --pp 2048,8192,32768,65536 --reps 1` against the running server; rows go to `results.jsonl` |
| Quality | first `pip install pyarrow` and `python3 bench/data/make_subsets.py /root/gbench/data`, then start a 4-lane server (`--max-context 4096 --max-concurrency 4`, see [`bench/runs/night1/night11.sh`](bench/runs/night1/night11.sh)) and run `python3 bench/geval.py --label NAME --suites gsm8k,mmlupro,mmlu,humaneval --conc 4` |
| Long context | `python3 bench/needle.py --label NAME --lengths 32000,72000,140000 --depths 0.1,0.5,0.9` |
| Agent turns | `python3 bench/turnprobe.py http://127.0.0.1:8080` |
| Perplexity of a kernel change | `ninferno-gemma4 MODEL score --tokens TOKENS --window 32768 --kv-dtype fp8 --context 32768 --dump F.bin`, then `bench/ppl.py F.bin` |
| One GEMM shape | `bench/lb.sh` or the linear bench directly (step 1) |
| Kernel mix | nsys, as in [`bench/prof_nv.sh`](bench/prof_nv.sh), summarised by [`bench/kerntop.py`](bench/kerntop.py). Decode runs in CUDA graphs: add `--cuda-graph-trace=node` |

[`bench/idle_gate.sh`](bench/idle_gate.sh) holds an overnight runner until the fast unit has served no request for N minutes, and while `/root/gbench/HOLD` exists. That way, tests never take the GPU from a user.

## Settings that mattered

| Setting | Why |
|---|---|
| `--drafter` matched to the target | fast: `assistant.q8full.ninferno` (Google's regular assistant) for the BF16-derived turbo. With the QAT assistant there, chat was 85 tok/s against 109. Max context: `qat-assistant.q40qat.ninferno`, the QAT-paired assistant losslessly in Q4_0, with acceptance 0.80–0.90 on code against 0.37 for the regular one, and decode 57 → 106 tok/s |
| `--sliding-kv-dtype fp8`, `NINFERNO_GEMMA4_EMBED_HOST=1` (fast) | FP8 sliding rings (0.77 GB) and the token table in host memory (1.4 GB): 202K with vision instead of 155K text-only, at the same speed; short agent turns 0.24 s instead of 0.28 s |
| `NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64` (fast) | the wide attention tile from 64 tokens: first token 11% sooner on agent turns that append about 900 tokens |
| `--vision` (fast) | images at about 290 tokens each; needs patch 12 with MTP |
| `--draft-tokens 7` | the verify runs 8 tokens and stays in the 8-token kernel bucket: chat +6%, decode after 8K +7–16% against 8 drafts. 12 and 15 drafts are slower |
| `NINFERNO_GEMMA4_SHARED_VALUES=2` | K-only FP8 KV for the ten k_eq_v full-attention layers, V rebuilt as unrope(K)/c in the kernel. This is what fits 155K (NVFP4) and 262K (Q4) |
| `NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1` | gate/up as a plain GEMM plus GELU·mul at every width. Prefill: 2.5K → 3.9K at 8K on NVFP4. Decode: +6–9% (NVFP4) and +10% (Q4). Our patch 01 adds the switch; its default (0) keeps the fused kernel |
| `--kv-dtype fp8` | half the KV of BF16, at unchanged quality scores |
| `--prefill-chunk 1024` (fast) / `512` (max context) | the context ceiling depends on the chunk: NVFP4 fits 92.7K at 2048, 155.8K at 1024 and 185.3K at 512. Chunk 1024 costs no prefill speed against 2048 |
| `--max-context 202112` (fast) | the largest that fits at chunk 1024 with vision, the regular assistant and both levers. Without `--vision` the same unit fits 262,144 |
