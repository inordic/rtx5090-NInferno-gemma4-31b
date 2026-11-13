#!/bin/bash
# phase 1: quantize T1 three ways on the GPU, KL against T1's own Q5 reference, production against stock's Q5
cd /root/gbench/n7
step() { echo "=== $(date +%T) $*"; }
R=/opt/models/n7/ref; Q=/opt/models/n7/T1
until [ "$(wc -l < refusal/A0.orhard.jsonl)" -ge 300 ]; do sleep 10; done
step "A0 battery done; chat corpus"; /opt/llama.cpp/.venv/bin/python -I mkcorpus.py 2>&1 | tail -1
step "stopping production"; systemctl stop gemma4-31b-fast; /root/gbench/srv.sh stop > /dev/null 2>&1
for i in $(seq 1 60); do curl -sf -m 2 http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
nvidia-smi --query-gpu=memory.used --format=csv,noheader
step "GPU quantization: rtn, mse, imw"
/opt/venv-cu/bin/python hlquant2.py $Q/bf16 $Q/q /root/gbench/n4/lila_scales.json $R/gemma-4-31b-it-heretic-ara.imatrix.gguf --device=cuda > quant2_T1.log 2>&1
grep -E "^layer 59|missing|DONE" quant2_T1.log | tail -3
grep -q "QUANTIZE2 DONE" quant2_T1.log || { step "PHASE1_FAILED (quantize)"; exit 1; }
step "converting the three (CPU) while the GPU records the reference"
( for m in rtn mse imw; do ./conv.sh $Q/q-$m $Q/bf16 /opt/models/n7/T1-$m.ninferno | tail -1; rm -rf $Q/q-$m; done > conv_T1.out 2>&1 ) &
./kld.sh ref $R/gemma-4-31b-it-heretic-ara.i1-Q5_K_M.ninferno /opt/models/n7/t1ref.kld
./kld.sh ref $R/gemma-4-31B-it-UD-Q5_K_XL.ninferno /opt/models/n7/stockref.kld
./kld.sh run /opt/models/dl2/turbo-qkv-vision-nvfp4.r31.ninferno /opt/models/n7/stockref.kld A0_vs_stock
wait; cat conv_T1.out
for m in rtn mse imw; do ./kld.sh run /opt/models/n7/T1-$m.ninferno /opt/models/n7/t1ref.kld T1-${m}_vs_t1; done
for m in rtn imw; do ./kld.sh run /opt/models/n7/T1-$m.ninferno /opt/models/n7/t1ref.kld T1-${m}_vs_t1_exact exact; done
./kld.sh run /opt/models/n7/T1-rtn.ninferno /opt/models/n7/stockref.kld T1-rtn_vs_stock
step "PHASE1_DONE"
