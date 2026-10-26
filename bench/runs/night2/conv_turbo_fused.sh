#!/bin/bash
# conv_turbo_fused: LilaRest NVFP4-turbo again, with each layer's query/key/value packed into one plane
# under the query's divisor (NINFERNO_CONVERT_NORMED_QKV_DIVISOR=1), like production's QKV. No GPU.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
until grep -q "GET DONE" get_turbo_src.log 2>/dev/null; do sleep 20; done
cp /opt/NInferno/build/apps/ninferno-convert /opt/NInferno/build/apps-final5/ninferno-convert-qkv
step "converting (fused QKV)"
cd /opt/models/dl3 && NINFERNO_CONVERT_NORMED_QKV_DIVISOR=1 nice -n 19 ionice -c3 /opt/NInferno/build/apps-final5/ninferno-convert-qkv \
    --model /opt/models/dl3/turbo-src --recipe gemma4_31b_nvfp4 --components text --out /opt/models/dl3/turbo-qkv-nvfp4.r31.ninferno 2>&1 \
    | grep -vE "^resource" | tail -4
ls -la /opt/models/dl3/turbo-qkv-nvfp4.r31.ninferno && python3 - <<"PY"
import json
d = json.load(open("/opt/models/dl3/turbo-qkv-nvfp4.r31.ninferno.conversion.json"))
print("formats", d["formats"])
for m in d["methods"]:
    p = m.get("parameters") or []
    if m.get("format") == "nvfp4" and any(x.startswith("text/layers/0/attention") or x.startswith("text/layers/5/attention") for x in p):
        print("  ", m["object"], [x.split("/", 3)[-1] for x in p], m.get("shape"))
PY
rm -rf /opt/models/dl3/turbo-src
step "CONV DONE"
