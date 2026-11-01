#!/bin/bash
# n4_prep: night 4's model preparation, CPU and network only (the GPU and the fast unit untouched).
#  1. waits for 125 GB free on / (cleanup_n4.sh), until 23:30 CET; without it the Humanlike tests skip.
#     Peak use is about 121 GB: downloads 81, the NVFP4 checkpoint 20 and its artifact 20; the GGUF
#     converts after the NVFP4 checkpoint is removed;
#  2. downloads LessThanThreeAI/Gemma-4-31B-Humanlike-Chat (BF16) and the GGUF repo's Q4_K_M, checks SHA256;
#  3. hlquant.py: a ModelOpt-style NVFP4 checkpoint of the BF16 weights with LilaRest's activation scales;
#  4. converts it as production's model was converted (BUILD 3b: Q/K/V packed, Google's vision tower),
#     and the Q4_K_M GGUF as the 262K unit's model was (gemma4_published, text).
cd /root/gbench/n4
step() { echo "=== $(date +%T) $*"; }
H=/opt/models/hl
CONV=/opt/NInferno/build/apps/ninferno-convert
PY=/opt/llama.cpp/.venv/bin/python
R=LessThanThreeAI/Gemma-4-31B-Humanlike-Chat
RG=LessThanThreeAI/Gemma-4-31B-Humanlike-Chat-GGUF
QF=Gemma-4-31B-Humanlike-Chat-Q4_K_M.gguf
deadline=$(date -u -d "2026-11-01 22:30" +%s)
until [ "$(df -BG --output=avail / | tail -1 | tr -dc 0-9)" -ge 125 ]; do
    [ "$(date -u +%s)" -ge $deadline ] && { step "no room: under 125 GB free at 23:30 CET"; step "PREP_SKIPPED"; exit 1; }
    sleep 60
done
step "room: $(df -h --output=avail / | tail -1) free"
mkdir -p $H/bf16 $H/gguf
fetch() {  # repo dest files...
    local repo=$1 dest=$2; shift 2
    for f in "$@"; do
        mkdir -p "$dest/$(dirname "$f")"
        for i in 1 2 3 4 5; do
            curl -sL -C - --fail --retry 10 --retry-delay 10 -o "$dest/$f" "https://huggingface.co/$repo/resolve/main/$f" && break
            sleep 15
        done
        echo "got $f $(stat -c%s "$dest/$f")"
    done
}
files=$(curl -s "https://huggingface.co/api/models/$R/tree/main" | python3 -c \
    'import json,sys; [print(f["path"]) for f in json.load(sys.stdin) if f["type"] == "file"]')
step "downloading the BF16 repo and the Q4_K_M GGUF"
fetch $RG $H/gguf $QF SHA256SUMS > dl_gguf.log 2>&1 &
fetch $R $H/bf16 $files > dl_bf16.log 2>&1
wait
cat dl_bf16.log dl_gguf.log
step "checking SHA256"
(cd $H/bf16 && grep -E "safetensors|tokenizer|config" SHA256SUMS | sha256sum -c --quiet) && echo "BF16 SHA OK" || step "BF16 SHA FAILED"
(cd $H/gguf && grep -F "$QF" SHA256SUMS | sed "s#[^ ]*$QF#$QF#" | sha256sum -c --quiet) && echo "GGUF SHA OK" || echo "GGUF SHA FAILED"

nvfp4() {  # the Humanlike NVFP4 artifact; any failure says NVFP4_FAILED and returns
grep -q "BF16 SHA OK" prep.out 2>/dev/null || { step "NVFP4_FAILED (BF16 checksum)"; return; }
until [ -s lila_scales.json ]; do sleep 30; done
step "quantizing to NVFP4 (ModelOpt layout, LilaRest's activation scales)"
OMP_NUM_THREADS=4 nice -n 15 $PY /root/gbench/hlquant.py quantize $H/bf16 $H/nvfp4-src lila_scales.json > quant.log 2>&1
tail -2 quant.log; grep -q "QUANTIZE DONE" quant.log || { step "NVFP4_FAILED (quantize)"; return; }
# the checkpoint reads as LilaRest's: its config (Gemma4ForCausalLM + ModelOpt NVFP4), Humanlike's tokenizer files
curl -sL -o $H/nvfp4-src/config.json https://huggingface.co/LilaRest/gemma-4-31B-it-NVFP4-turbo/resolve/main/config.json
cp $H/bf16/generation_config.json $H/bf16/tokenizer.json $H/bf16/tokenizer_config.json $H/bf16/chat_template.jinja \
   $H/bf16/processor_config.json $H/nvfp4-src/
step "converting the NVFP4 checkpoint (BUILD 3b: Q/K/V packed, Google's vision tower)"
NINFERNO_CONVERT_NORMED_QKV_DIVISOR=1 nice -n 15 $CONV --model $H/nvfp4-src --recipe gemma4_31b_nvfp4 \
    --source vision=/opt/models/dl3/it-mmproj/mmproj-BF16.gguf --components text,vision --threads 4 \
    --out $H/humanlike-qkv-vision-nvfp4.r31.ninferno > conv_nvfp4.log 2>&1
tail -3 conv_nvfp4.log
python3 -c "
import json; d = json.load(open('$H/humanlike-qkv-vision-nvfp4.r31.ninferno.conversion.json'))
print('formats', d.get('formats')); print('components', [c.get('name', c) if isinstance(c, dict) else c for c in d.get('components', [])])" \
    && ls -la $H/humanlike-qkv-vision-nvfp4.r31.ninferno && step "NVFP4_DONE" || step "NVFP4_FAILED (convert)"
}
nvfp4
rm -rf $H/nvfp4-src
step "GGUF: converting Q4_K_M (gemma4_published, text)"
( nice -n 15 $CONV --model $H/gguf/$QF --components text --threads 4 --out $H/humanlike-q4km.ninferno > conv_gguf.log 2>&1 \
    && ls -la $H/humanlike-q4km.ninferno && echo GGUF_CONVERT_DONE || echo GGUF_CONVERT_FAILED ) >> conv_gguf.log 2>&1
grep -E "GGUF_CONVERT_(DONE|FAILED)" conv_gguf.log | tail -1
df -h / | tail -1
step "PREP_DONE"
