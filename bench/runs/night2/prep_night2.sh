#!/bin/bash
# prep_night2: artifacts for night 2, at idle CPU and IO priority (no GPU): the QAT Q4 and QAT-NVFP4
# targets with their vision towers, and the Huihui QAT-abliterated NVFP4 (uncensored candidate).
C="nice -n 19 ionice -c3 /opt/NInferno/build/apps/ninferno-convert"
step() { echo "=== $(date +%T) $*"; }
g() { nice -n 19 curl -sL -C - --fail --retry 10 --retry-delay 10 -o "$2" "https://huggingface.co/$1/resolve/main/$3"; }
step "mmproj for the QAT-UD Q4 GGUF"
cd /opt/models/dl && g unsloth/gemma-4-31B-it-qat-GGUF mmproj-BF16.gguf mmproj-BF16.gguf && ls -la mmproj-BF16.gguf
step "convert QAT-UD Q4 + vision"
$C --model /opt/models/dl/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf --components text,vision --out /opt/models/dl/qat-ud-q4-vision.ninferno 2>&1 | grep -vE "^resource" | tail -3
ls -la /opt/models/dl/qat-ud-q4-vision.ninferno
step "convert QAT-NVFP4 + vision"
cd /opt/models/dl2 && $C --model /opt/models/dl2/qat-nvfp4 --recipe gemma4_31b_nvfp4 --components text,vision --out /opt/models/dl2/qat-nvfp4-vision.r31.ninferno 2>&1 | grep -vE "^resource" | tail -3
ls -la /opt/models/dl2/qat-nvfp4-vision.r31.ninferno
step "download Huihui QAT-abliterated NVFP4"
R=sakamakismile/Huihui-gemma-4-31B-it-qat-abliterated-NVFP4
D=/opt/models/dl3/huihui-qat-abl-nvfp4; mkdir -p $D; cd $D
for f in config.json generation_config.json processor_config.json tokenizer.json tokenizer_config.json model.safetensors; do g $R $f $f && echo "got $f $(stat -c%s $f)"; done
step "convert Huihui QAT-abliterated NVFP4 (text)"
cd /opt/models/dl3 && $C --model $D --recipe gemma4_31b_nvfp4 --components text --out /opt/models/dl3/huihui-qat-abl-nvfp4.r31.ninferno 2>&1 | grep -vE "^resource" | tail -3
ls -la /opt/models/dl3/huihui-qat-abl-nvfp4.r31.ninferno
step "PREP DONE"
