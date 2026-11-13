#!/bin/bash
# night-8 runner: both downloads done -> production stopped -> both quantized (imw) on the GPU -> converted ->
# battery on each -> production back. One copy only.
exec 9> /root/gbench/n8/n8.lock; flock -n 9 || { echo "n8 already running"; exit 1; }
cd /root/gbench/n8
step() { echo "=== $(date +%T) $*"; }
until grep -qE "DL_(DONE|FAILED)" dl_M2.out && grep -qE "DL_(DONE|FAILED)" dl_A2.out; do sleep 30; done
tail -n 1 dl_M2.out dl_A2.out
step "stopping production"; systemctl stop gemma4-31b-fast; /root/gbench/srv.sh stop > /dev/null 2>&1
for i in $(seq 1 60); do curl -sf -m 2 http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
for N in M2 A2; do
    grep -q "DL_DONE $N" dl_$N.out || { step "$N: download failed, skipped"; continue; }
    step "$N: imw quantization on the GPU"
    (cd /root/gbench/n7 && /opt/venv-cu/bin/python hlquant2.py /opt/models/n8/$N/bf16 /opt/models/n8/$N/q /root/gbench/n4/lila_scales.json /opt/models/n8/ref/$N.imatrix.gguf --device=cuda --methods=imw) > quant_$N.log 2>&1
    grep -E "imatrix missing|DONE" quant_$N.log | tail -2
done
for N in M2 A2; do
    [ -d /opt/models/n8/$N/q-imw ] || continue
    step "$N: converting"; /root/gbench/n7/conv.sh /opt/models/n8/$N/q-imw /opt/models/n8/$N/bf16 /opt/models/n8/$N-imw.ninferno | tail -1
    grep -q "CONV_DONE" /opt/models/n8/$N-imw.ninferno.convlog 2>/dev/null; rm -rf /opt/models/n8/$N/q-imw
    [ -s /opt/models/n8/$N-imw.ninferno.conversion.json ] && { step "$N: battery"; ./battery8.sh $N /opt/models/n8/$N-imw.ninferno; grep -E "BATTERY_DONE|DIED" $N.txt | tail -1; }
done
step "production back"; /root/gbench/srv.sh stop > /dev/null 2>&1; /root/gbench/deploy_fast.sh
for i in $(seq 1 60); do curl -sf -m 5 http://127.0.0.1:8080/v1/models > /dev/null && break; sleep 5; done
step "N8_DONE: production $(systemctl is-active gemma4-31b-fast)"
