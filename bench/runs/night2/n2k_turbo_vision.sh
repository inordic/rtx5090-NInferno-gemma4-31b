#!/bin/bash
# n2k_turbo_vision: the recommended build end to end - LilaRest NVFP4-turbo with Q/K/V packed and a vision
# tower, Google's regular assistant (7 drafts), FP8 sliding rings, the token table in host memory and the
# wide prefill tile from 64 tokens, on apps-final8 (patches 10-13). K0 is production as deployed, for the
# prefill baseline in the same session; K1 the build with --vision at its ceiling; K2 without vision.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
until grep -qE "CONV2 (DONE|FAILED)" conv_turbo_vision.log 2>/dev/null && grep -qE "BUILD_final8_(OK|FAILED)" build_aside_final8.out 2>/dev/null; do sleep 20; done
grep -q "CONV2 DONE" conv_turbo_vision.log && grep -q "BUILD_final8_OK" build_aside_final8.out || { step "prerequisite failed"; step "N2K DONE"; exit 1; }
./idle_gate_any.sh 5
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
S=/opt/NInferno/build/apps-final8/ninferno-serve
sed "s#^S=.*#S=$S#" srv.sh > srv_final8.sh; chmod +x srv_final8.sh
TV=/opt/models/dl3/turbo-qkv-vision-nvfp4.r31.ninferno
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
DR=/opt/models/dl2/assistant.q8full.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
LV="--kv-dtype fp8 --sliding-kv-dtype fp8"
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_SHARED_VALUES=2

step "K0 production as deployed (apps-final7, 155K, QAT drafter): prefill baseline"
./srv_final7.sh start K0-production $QN --max-context 155000 --prefill-chunk 1024 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 7 \
    --chat-template /opt/models/templates/gemma4-agent.jinja && \
    python3 gbench.py --label "K0 production apps-final7 155K" --pp 2048,8192,32768,65536 --reps 2
./srv.sh stop

export NINFERNO_GEMMA4_EMBED_HOST=1 NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64
ceil() { $S $TV --plan $LV --max-context 262144 --prefill-chunk 1024 --spec mtp --drafter $DR --draft-tokens 7 "$@" 2>&1 \
         | grep -oE "with up to [0-9]+ tokens" | head -1 | grep -oE "[0-9]+"; }
CV=$(ceil --vision); [ -z "$CV" ] && CV=262144
CT=$(ceil); [ -z "$CT" ] && CT=262144
echo "CEIL turbo-qkv-vision levers regular drafter: vision $CV, text $CT"

step "K1 recommended build with --vision at $CV"
./srv_final8.sh start K1-turbo-vision $TV --vision --max-context $CV --prefill-chunk 1024 $LV --spec mtp --drafter $DR --draft-tokens 7 \
    --chat-template /opt/models/templates/gemma4-agent.jinja && \
    python3 visiontest.py --label "K1 turbo-qkv-vision ctx$CV" && \
    python3 gbench.py --label "K1 turbo-qkv-vision ctx$CV" --pp 2048,8192,32768,65536 --reps 2 && \
    python3 editbench.py --label "K1 turbo-qkv-vision" && \
    python3 turnprobe.py http://127.0.0.1:8080 && \
    python3 agenttest.py --label "K1 turbo-qkv-vision" --runs 1 && \
    python3 needle.py --label "K1 turbo-qkv-vision needles" --lengths 140000,$(( CV * 9 / 10 )) --depths 0.5 && \
    python3 geval.py --label "K1-turbo-qkv-vision" --suites mmlupro --conc 1
grep -hE "Engine failed|ERROR|FATAL" logs/K1-turbo-vision.log | sed -E "s/^.*(ninferno: |ERROR +|FATAL +)//" | cut -c1-300 | head -3
./srv.sh stop

step "K2 the same without --vision at $CT"
./srv_final8.sh start K2-turbo-text $TV --max-context $CT --prefill-chunk 1024 $LV --spec mtp --drafter $DR --draft-tokens 7 \
    --chat-template /opt/models/templates/gemma4-agent.jinja && \
    python3 gbench.py --label "K2 turbo-qkv text ctx$CT" --pp 8192 --tg "" --reps 1 && \
    python3 needle.py --label "K2 turbo-qkv text needle" --lengths $(( CT * 9 / 10 )) --depths 0.5
./srv.sh stop
step "restoring the fast unit"
./deploy_fast.sh
step "N2K DONE"
