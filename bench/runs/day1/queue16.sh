#!/bin/bash
# queue16: after queue15 - where does an MTP K=8 decode step go? nsys over the serve process while
# it answers one 512-token chat request (NVFP4 + split + INT8, shared2, 16K context).
cd /root/gbench
until grep -qE "QUEUE15 DONE|PV8 BUILD FAILED" queue15.log 2>/dev/null; do sleep 15; done
step() { echo "=== $(date +%T) $*"; }
./srv.sh stop; sleep 2
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=64 NINFERNO_GEMMA4_SHARED_VALUES=2
step "profile decode: serve under nsys, one 512-token chat"
cd /opt/models
PATH=/usr/local/cuda-13.2/bin:$PATH nsys profile -f true -o /root/gbench/prof_decode --trace=cuda --sample=none --cpuctxsw=none --duration=75 \
    /opt/NInferno/build/apps-a8/ninferno-serve $QN --host 0.0.0.0 --port 8080 --model-id prof --max-context 16384 --kv-dtype fp8 \
    --spec mtp --drafter $DQ --draft-tokens 8 > /root/gbench/logs/prof_decode.log 2>&1 &
cd /root/gbench
for i in $(seq 1 120); do grep -q "listening on" logs/prof_decode.log && break; sleep 1; done
python3 - <<"PY"
import json, time, urllib.request
body = {"model": "x", "messages": [{"role": "user", "content": "Write a detailed explanation of how a hash map works, with a Python implementation."}], "max_tokens": 512, "temperature": 0}
for _ in range(2):
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})))
    print("chat", r["usage"], f"{time.time() - t:.2f}s")
PY
wait
grep -E "req#.*done" logs/prof_decode.log | sed -E "s/^.*INFO +//" | cut -c1-260
nsys stats --report cuda_gpu_kern_sum --format csv -o /root/gbench/prof_decode /root/gbench/prof_decode.nsys-rep > /dev/null 2>&1
python3 /root/gbench/kerntop.py /root/gbench/prof_decode_cuda_gpu_kern_sum.csv | head -30
step "QUEUE16 DONE"
