# Build: from a container with an RTX 5090 Laptop GPU to both profiles serving

This is the route we took. The paths are ours:
- **Engine:** `/opt/NInferno`, built in `/opt/NInferno/build`. Production runs a copy of the binaries in `build/apps-final3`, so later builds in the tree don't touch it.
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

```bash
git clone https://github.com/Brontir/NInferno.git /opt/NInferno
cd /opt/NInferno
git checkout -b kvshare-main 9895ebc62e2bdfb367083a72cb2c4a714cf41542
git am /path/to/this/repo/patches/ninferno/*.patch       # patches 01-07, see patches/README.md
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
PATH=/usr/local/cuda-13.2/bin:$PATH ninja -C build ninferno-serve ninferno-gemma4 ninferno ninferno-convert
mkdir -p build/apps-final3 && cp build/apps/ninferno-serve build/apps/ninferno-gemma4 build/apps/ninferno build/apps-final3/
```

The full build takes a long time; the kernels are many. Add `-DNINFERNO_BUILD_BENCHMARKS=ON` and `ninja -C build bench/ninferno_linear_bench` if you want the kernel microbenchmark used in [`bench/lb.sh`](bench/lb.sh).

**Check:**
- `git log --oneline -1` shows `q6 head: the sliced 8-column tile for the Gemma 4 31B head at T 6..8`, our patch 07.
- `build/apps-final3/ninferno-serve --version` prints `NInferno 0.1.0 (built for sm_120, CUDA 13.2)` and lists the GPU.

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

### 3a. The drafter (both profiles)

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

### 3b. The fast profile's target: QAT-NVFP4

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

### 3c. The max-context profile's target: QAT-UD Q4

```bash
cd /opt/models/dl && curl -sL -C - --fail --retry 10 -o gemma-4-31B-it-qat-UD-Q4_K_XL.gguf \
    https://huggingface.co/unsloth/gemma-4-31B-it-qat-GGUF/resolve/main/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf
/opt/NInferno/build/apps/ninferno-convert --model /opt/models/dl/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf
```

Without `--out`, the converter writes to `.ninferno-cache/gemma-4-31B-it-qat-UD-Q4_K_XL.gemma4_published.ninferno` next to the GGUF. Despite the name, every matrix in this GGUF is Q4_0 on the QAT grid.

**Check:** the artifact is about 17.3 GB, and its `.conversion.json` lists `ggml_q4_0: 241`.

### 3d. Optional: uncensored (Heretic, QAT-aware Q4_0)

[`bench/models/heretic.sh`](bench/models/heretic.sh) downloads `llmfan46/gemma-4-31B-it-qat-q4_0-unquantized-uncensored-heretic` and converts it with the same QAT-aware encoder. Then convert the GGUF as in 3c. It runs on the max-context profile's config at about 104 tok/s; its quality suite has not been run yet.

## 4. Serve

1. Copy [`config/systemd/gemma4-31b-fast.service`](config/systemd/gemma4-31b-fast.service) and [`config/systemd/gemma4-31b-262k.service`](config/systemd/gemma4-31b-262k.service) to `/etc/systemd/system/`. Set the paths if yours differ.
2. Run `systemctl daemon-reload && systemctl start gemma4-31b-fast`.

The two units conflict, so starting one stops the other. They are not enabled at boot; enable the one you want with `systemctl enable`.

The first start takes about 25 s: weights, plan and CUDA graphs. The log line `plan | rung 3 of 3: 155,000 tokens x 1 context` confirms the context.

**Check**, in order:
1. `curl -s localhost:8080/v1/models` shows `max_model_len 155000` and `"config": "mtp K=7"`.
2. [`bench/verify.py`](bench/verify.py) passes: models, chat, streaming, and three agent-style turns with prefix reuse. The second and third turns should report `cache` around 17,600 and take about 0.5 s.
3. [`bench/gbench.py`](bench/gbench.py) `--pp 2048,8192,32768,65536` lands near the numbers in [`results/results.jsonl`](results/results.jsonl), label `C3 candidate final3 K7`.
4. [`bench/needle.py`](bench/needle.py) `--lengths 32000,72000,140000 --depths 0.1,0.5,0.9` scores 9/9.

[`bench/deploy_fast.sh`](bench/deploy_fast.sh) stops anything on 8080 and brings the fast unit up again. Our overnight runners call it after every test batch.

## 5. Optional: the NInfer web UI

The UI is iNordic's NInfer web UI, a Node app: release 0.0.9, commit `ef4ebaa`. It isn't in this repository. Put the app in `/opt/ninfer-ui/app` and its bundled Node 22 runtime in `/opt/ninfer-ui/runtime`, then:

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

**Check:** `curl http://<host-ip>:8081/health` shows `"model":"gemma4-31b-fast","max_context":155000`.

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
| `--drafter qat-assistant.q40qat.ninferno` | the QAT-paired assistant, losslessly in Q4_0: acceptance 0.80–0.90 on code, against 0.37 for the regular assistant. Decode 57 → 106 tok/s |
| `--draft-tokens 7` | the verify runs 8 tokens and stays in the 8-token kernel bucket: chat +6%, decode after 8K +7–16% against 8 drafts. 12 and 15 drafts are slower |
| `NINFERNO_GEMMA4_SHARED_VALUES=2` | K-only FP8 KV for the ten k_eq_v full-attention layers, V rebuilt as unrope(K)/c in the kernel. This is what fits 155K (NVFP4) and 262K (Q4) |
| `NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1` | gate/up as a plain GEMM plus GELU·mul at every width. Prefill: 2.5K → 3.9K at 8K on NVFP4. Decode: +6–9% (NVFP4) and +10% (Q4). Our patch 01 adds the switch; its default (0) keeps the fused kernel |
| `--kv-dtype fp8` | half the KV of BF16, at unchanged quality scores |
| `--prefill-chunk 1024` (fast) / `512` (max context) | the context ceiling depends on the chunk: NVFP4 fits 92.7K at 2048, 155.8K at 1024 and 185.3K at 512. Chunk 1024 costs no prefill speed against 2048 |
| `--max-context 155000` | the largest that fits at chunk 1024. At 100K the plan keeps two cached states on the device, and each cached agent turn is 80–100 ms faster |
