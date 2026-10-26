#!/bin/bash
# prep_night2c: two more NVFP4 targets for the knowledge-gap question, at idle priority (no GPU):
# LilaRest NVFP4-turbo (made from Google's BF16 weights) and yasu-oh QAT NVFP4 (QAT weights, an
# error-minimising recipe). Each is converted like production (Q8 table, Q6 head, text only),
# then its download is removed.
C="nice -n 19 ionice -c3 /opt/NInferno/build/apps/ninferno-convert"
T=/opt/models/dl2/qat-nvfp4/chat_template.jinja
step() { echo "=== $(date +%T) $*"; }
get() {  # get <repo> <dir>
    mkdir -p $2 && cd $2
    for f in $(curl -s "https://huggingface.co/api/models/$1/tree/main" | python3 -c "import json,sys;[print(x[\"path\"]) for x in json.load(sys.stdin) if x[\"type\"]==\"file\"]"); do
        nice -n 19 curl -sL -C - --fail --retry 10 --retry-delay 10 -o "$f" "https://huggingface.co/$1/resolve/main/$f" && echo "got $f $(stat -c%s $f)"
    done
}
conv() {  # conv <dir> <out>
    local extra=""; [ -f $1/chat_template.jinja ] || extra="--resource chat_template.jinja=$T"
    cd /opt/models/dl3 && $C --model $1 --recipe gemma4_31b_nvfp4 --components text $extra --out $2 2>&1 | grep -vE "^resource" | tail -3
    ls -la $2 && rm -rf $1
}
step "LilaRest NVFP4-turbo"; get LilaRest/gemma-4-31B-it-NVFP4-turbo /opt/models/dl3/turbo-src
conv /opt/models/dl3/turbo-src /opt/models/dl3/turbo-nvfp4.r31.ninferno
step "yasu-oh QAT NVFP4"; get yasu-oh/gemma-4-31B-it-qat-NVFP4 /opt/models/dl3/yasu-src
conv /opt/models/dl3/yasu-src /opt/models/dl3/yasu-qat-nvfp4.r31.ninferno
step "PREP3 DONE"
