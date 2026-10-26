#!/bin/bash
# conv_turbo_vision: LilaRest NVFP4-turbo with Q/K/V packed under one divisor (patch 13) and a vision
# tower: its own if the checkpoint has one, else the projector of Google's original IT model. No GPU.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
until grep -q "GET DONE" prep_turbo_vision.log 2>/dev/null; do sleep 20; done
C="nice -n 19 ionice -c3 /opt/NInferno/build/apps-final5/ninferno-convert-qkv"
OUT=/opt/models/dl3/turbo-qkv-vision-nvfp4.r31.ninferno
cd /opt/models/dl3
if grep -q "vision tensors in checkpoint: [1-9]" /root/gbench/prep_turbo_vision.log; then
    step "converting: turbo's own vision tower"
    NINFERNO_CONVERT_NORMED_QKV_DIVISOR=1 $C --model /opt/models/dl3/turbo-src --recipe gemma4_31b_nvfp4 \
        --components text,vision --out $OUT 2>&1 | grep -vE "^resource" | tail -4
else
    step "converting: Google's original IT projector"
    NINFERNO_CONVERT_NORMED_QKV_DIVISOR=1 $C --model /opt/models/dl3/turbo-src --recipe gemma4_31b_nvfp4 \
        --source vision=/opt/models/dl3/it-mmproj/mmproj-BF16.gguf --components text,vision --out $OUT 2>&1 | grep -vE "^resource" | tail -4
fi
if [ -f $OUT ]; then
    ls -la $OUT
    python3 - <<"PY"
import json
d = json.load(open("/opt/models/dl3/turbo-qkv-vision-nvfp4.r31.ninferno.conversion.json"))
print("formats", d["formats"], "components", list(d["components"].keys()))
qkv = [m for m in d["methods"] if any(x.endswith("attention/query") for x in (m.get("parameters") or []))]
print("query objects", len(qkv), "packed with key:", sum(1 for m in qkv if any(x.endswith("attention/key") for x in m["parameters"])))
print("vision objects", sum(1 for m in d["methods"] if any(x.startswith("vision/") for x in (m.get("parameters") or []))))
PY
    rm -rf /opt/models/dl3/turbo-src
    step "CONV2 DONE"
else
    step "CONV2 FAILED"
fi
