#!/bin/bash
# n7 prep: one BF16 Gemma 4 31B fine-tune -> production-layout NVFP4 (hlquant, LilaRest's activation scales)
# -> ninferno artifact as production's (BUILD 3b: Q/K/V packed, Google's vision tower). CPU and network only.
#   prep.sh REPO NAME   ->  /opt/models/n7/NAME.qkv-vision-nvfp4.ninferno ; log n7/prep_NAME.out
R=$1; N=$2; B=/opt/models/n7/$N; OUT=/opt/models/n7/$N.qkv-vision-nvfp4.ninferno
CONV=/opt/NInferno/build/apps/ninferno-convert; PY=/opt/llama.cpp/.venv/bin/python
step() { echo "=== $(date +%T) $*"; }
mkdir /opt/models/n7/lock-$N 2>/dev/null || { echo "=== $N is already being prepared or done"; exit 0; }
mkdir -p $B/bf16; cd $B/bf16
step "listing $R"
curl -s "https://huggingface.co/api/models/$R/tree/main?recursive=1" > ../tree.json
python3 -c "
import json; t=json.load(open('../tree.json'))
if isinstance(t, dict): print('ERR', t); raise SystemExit(1)
for x in t:
    if x['type']=='file' and not x['path'].endswith(('.gguf','.png','.jpg','.md')) and '/' not in x['path']:
        print(x['path'], (x.get('lfs') or {}).get('oid',''))" > ../files.txt || { cat ../files.txt; step "PREP_FAILED (listing)"; exit 1; }
step "downloading $(wc -l < ../files.txt) files"
while read f sha; do
    for i in 1 2 3 4 5; do curl -sL -C - --fail --retry 10 --retry-delay 10 -o "$f" "https://huggingface.co/$R/resolve/main/$f" && break; sleep 15; done
    if [ -n "$sha" ]; then echo "$sha  $f" >> ../sha.txt; fi
done < ../files.txt
du -sh .
step "checking SHA256"
sha256sum -c --quiet ../sha.txt && echo "SHA OK" || { step "PREP_FAILED (sha)"; exit 1; }
python3 -c "
import json; w=json.load(open('model.safetensors.index.json'))['weight_map']
n=[k for k in w if k.startswith('model.language_model.layers.')]; print('language_model tensors', len(n), 'all', len(w))
raise SystemExit(0 if len(n) > 500 else 1)" || { step "PREP_FAILED (tensor names differ from Gemma4ForConditionalGeneration)"; exit 1; }
step "quantizing to NVFP4 (ModelOpt layout, LilaRest's activation scales)"
OMP_NUM_THREADS=6 nice -n 15 $PY /root/gbench/hlquant.py quantize $B/bf16 $B/nvfp4-src /root/gbench/n4/lila_scales.json > ../quant.log 2>&1
tail -2 ../quant.log; grep -q "QUANTIZE DONE" ../quant.log || { step "PREP_FAILED (quantize)"; exit 1; }
curl -sL -o $B/nvfp4-src/config.json https://huggingface.co/LilaRest/gemma-4-31B-it-NVFP4-turbo/resolve/main/config.json
for f in generation_config.json tokenizer.json tokenizer_config.json chat_template.jinja processor_config.json special_tokens_map.json; do
    [ -f $f ] && cp $f $B/nvfp4-src/ || echo "no $f in the source"; done
[ -f $B/nvfp4-src/chat_template.jinja ] || cp /opt/models/hl/bf16/chat_template.jinja $B/nvfp4-src/
step "converting (BUILD 3b: Q/K/V packed, Google's vision tower)"
NINFERNO_CONVERT_NORMED_QKV_DIVISOR=1 nice -n 15 $CONV --model $B/nvfp4-src --recipe gemma4_31b_nvfp4 \
    --source vision=/opt/models/dl3/it-mmproj/mmproj-BF16.gguf --components text,vision --threads 6 --out $OUT > ../conv.log 2>&1
tail -2 ../conv.log
python3 -c "import json; d=json.load(open('$OUT.conversion.json')); print('formats', d.get('formats'))" && ls -la $OUT && rm -rf $B/nvfp4-src && step "PREP_DONE $N" || step "PREP_FAILED (convert)"
