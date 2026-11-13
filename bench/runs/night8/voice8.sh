#!/bin/bash
# after n8.sh: sampled voice for M2 and A2 (each served with production's flags), then production back and its voice
exec 9> /root/gbench/n8/voice8.lock; flock -n 9 || { echo "voice8 already running"; exit 1; }
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
until grep -q N8_DONE n8/n8.out; do sleep 15; done
export NINFERNO_GEMMA4_SHARED_VALUES=2 NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_EMBED_HOST=1 NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64 NINFERNO_GEMMA4_SKIP_LAST_LAYER=1
PROD="--vision --max-context 203136 --prefill-chunk 1280 --kv-dtype fp8 --sliding-kv-dtype fp8 --spec mtp --drafter /opt/models/dl2/assistant.nvfp4.ninferno --draft-tokens 7 --chat-template /opt/models/templates/gemma4-agent.jinja --auto-tools-shared-prefix --auto-shared-prefix-requested"
systemctl stop gemma4-31b-fast; ./srv.sh stop > /dev/null 2>&1
for N in M2 A2; do
    [ -s /opt/models/n8/$N-imw.ninferno.conversion.json ] || continue
    SRV_BIN=/opt/NInferno-up/build/apps-up4/ninferno-serve ./srv_any.sh start V$N /opt/models/n8/$N-imw.ninferno $PROD > /dev/null || { step "$N DIED"; continue; }
    python3 -I n8/voice.py $N; ./srv.sh stop > /dev/null 2>&1
done
step "production back"; ./deploy_fast.sh
for i in $(seq 1 60); do curl -sf -m 5 http://127.0.0.1:8080/v1/models > /dev/null && break; sleep 5; done
python3 -I n8/voice.py A0
step "VOICE8_DONE: production $(systemctl is-active gemma4-31b-fast)"
