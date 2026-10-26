#!/bin/bash
# night2: after night1, behind a 15-minute idle gate - QAT Q4_0 prefill on the producer-warp W4A8 tile
# (NINFERNO_Q40_W4A8_WS=1) against ggml MMQ: lab prefill, 32K perplexity, nsys kernel mix.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
until grep -q "NIGHT1 DONE" night1.log 2>/dev/null; do sleep 30; done
step "waiting for 15 idle minutes on 8080"
./idle_gate.sh 15
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
QU=/opt/models/dl/.ninferno-cache/gemma-4-31B-it-qat-UD-Q4_K_XL.gemma4_published.ninferno
G=/opt/NInferno/build/apps-q40ws/ninferno-gemma4
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1
for ws in 0 1; do
    for p in 8192 32768; do
        printf "LAB q40ws=%s %s " $ws $p
        NINFERNO_Q40_W4A8_WS=$ws $G $QU bench --tokens /root/gbench/prompt128k.i32 --prompt $p --decode 4 --kv-dtype fp8 --context $((p + 2048)) --chunk 2048 2>&1 | grep -oE "\"prefill_tok_s\":[0-9.]+|\"decode_tok_s\":[0-9.]+" | tr "\n" " "; echo
    done
    step "score q4 q40ws=$ws window 32768"
    NINFERNO_Q40_W4A8_WS=$ws $G $QU score --tokens /root/gbench/ppl64k.i32 --window 32768 --kv-dtype fp8 --context 32768 --dump /root/gbench/score_q4_ws${ws}_32768.bin > /dev/null 2>&1
    python3 ppl.py /root/gbench/score_q4_ws${ws}_32768.bin
    step "profile q4 q40ws=$ws prefill 6.6K"
    NINFERNO_Q40_W4A8_WS=$ws nsys profile -f true -o /root/gbench/prof_q4_ws$ws --trace=cuda --sample=none --cpuctxsw=none \
        $G $QU bench --tokens /root/gbench/prompt128k.i32 --prompt 6656 --decode 4 --kv-dtype fp8 --context 8192 --chunk 2048 > /dev/null 2>&1
    nsys stats --report cuda_gpu_kern_sum --format csv -o /root/gbench/prof_q4_ws$ws /root/gbench/prof_q4_ws$ws.nsys-rep > /dev/null 2>&1
    python3 /root/gbench/kerntop.py /root/gbench/prof_q4_ws${ws}_cuda_gpu_kern_sum.csv | head -12
done
step "restoring the fast unit"
./deploy_fast.sh
step "NIGHT2 DONE"
