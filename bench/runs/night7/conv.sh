#!/bin/bash
# conv.sh CKPT_DIR BF16_DIR OUT: an hlquant(2) NVFP4 checkpoint -> ninferno artifact as production's (BUILD 3b)
C=$1; B=$2; OUT=$3; CONV=/opt/NInferno/build/apps/ninferno-convert
curl -sL -o $C/config.json https://huggingface.co/LilaRest/gemma-4-31B-it-NVFP4-turbo/resolve/main/config.json
for f in generation_config.json tokenizer.json tokenizer_config.json chat_template.jinja processor_config.json special_tokens_map.json; do
    [ -f $B/$f ] && cp $B/$f $C/; done
[ -f $C/chat_template.jinja ] || cp /opt/models/hl/bf16/chat_template.jinja $C/
NINFERNO_CONVERT_NORMED_QKV_DIVISOR=1 nice -n 15 $CONV --model $C --recipe gemma4_31b_nvfp4 \
    --source vision=/opt/models/dl3/it-mmproj/mmproj-BF16.gguf --components text,vision --threads 6 --out $OUT > $OUT.convlog 2>&1
python3 -c "import json; d=json.load(open('$OUT.conversion.json')); print('formats', d.get('formats'))" && ls -la $OUT && echo "CONV_DONE $OUT" || { tail -3 $OUT.convlog; echo "CONV_FAILED $OUT"; }
