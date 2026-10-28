#!/bin/bash
# night3_d BIN_DIR: after night3_b, behind the idle gate. The proposed fast unit end to end: BIN_DIR's
# ninferno-serve at production flags plus patch 16's switches and NInferno's default host cache
# (verify, tool calls, vision, agent loop, the cache replay, pinned memory); and the replay without
# the tools candidate (C5) to see whether it is needed. Then the fast unit back.
BIN=${1:?bin dir}/ninferno-serve
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
until grep -q "N3B DONE" night3_b.out 2>/dev/null; do sleep 30; done
./idle_gate_any.sh 5
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
export NINFERNO_GEMMA4_SHARED_VALUES=2 NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_EMBED_HOST=1 NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64
M=/opt/models/dl2/turbo-qkv-vision-nvfp4.r31.ninferno
DR=/opt/models/dl2/assistant.q8full.ninferno
PROD="--vision --max-context 202112 --prefill-chunk 1024 --kv-dtype fp8 --sliding-kv-dtype fp8 --spec mtp --drafter $DR --draft-tokens 7 --chat-template /opt/models/templates/gemma4-agent.jinja"
CACHE="--host-state-slots 18 --host-kv-mib 8192"
efail() { grep -hE "Engine failed" logs/$1.log | sed -E "s/^.*ninferno: //" | cut -c1-300 | head -2; }

step "D1 proposed unit: $BIN + flags + cache"
SRV_BIN=$BIN ./srv_any.sh start D1-proposed $M $PROD --auto-tools-shared-prefix --auto-shared-prefix-requested $CACHE | grep -E "context cache|plan \| rung"
pid=$(pgrep -f "apps[^ ]*/ninferno-serve" | head -1)
python3 verify.py 2>&1 | tail -3 && python3 textcalltest.py 2>&1 | tail -1 && \
    python3 visiontest.py --label "D1 proposed" 2>&1 | tail -1 && python3 agenttest.py --label "D1 proposed" --runs 1 2>&1 | tail -1 && \
    python3 cachereplay.py --label "D1 proposed" --slots 12 > replay_D1.log 2>&1
tail -1 replay_D1.log
echo "serve RSS $(ps -o rss= -p $pid | awk "{printf \"%.1f GiB\", \$1/1048576}"), container $(free -g | awk "/Mem:/{print \$3\" GiB used, \"\$7\" GiB available\"}")"
efail D1-proposed
./srv.sh stop

step "C5 replay: requested evidence and the cache, no tools candidate"
SRV_BIN=$BIN ./srv_any.sh start C5-replay $M $PROD --auto-shared-prefix-requested $CACHE > /dev/null && \
    python3 cachereplay.py --label "C5 --auto-shared-prefix-requested $CACHE" --slots 12 > replay_C5.log 2>&1
tail -1 replay_C5.log; efail C5-replay
./srv.sh stop
step "restoring the fast unit"
./deploy_fast.sh
step "N3D DONE"
