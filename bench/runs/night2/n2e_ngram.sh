#!/bin/bash
# n2e_ngram: night 2, batch E - n-gram copy rounds beside MTP on agent-style edits: off, the default
# (min match 12, 15 tokens) and lower thresholds, on the production config; short chat as a check
# that nothing else moves.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
until grep -q "N2F DONE" n2f_speed.log 2>/dev/null; do sleep 30; done
./idle_gate_any.sh 5
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_SHARED_VALUES=2
for cfg in "off:--ngram-draft-tokens 0" "default:" "m8:--ngram-min-match 8" "m4:--ngram-min-match 4" "m6:--ngram-min-match 6"; do
    name=${cfg%%:*}; args=${cfg#*:}
    step "ngram $name ($args)"
    ./srv_final5.sh start E-$name $QN --max-context 155000 --prefill-chunk 1024 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 7 $args && \
        python3 editbench.py --label "E ngram $name" && \
        python3 gbench.py --label "E ngram $name chat" --pp "" --tg "" --reps 1
    grep -hoE "copies [0-9]+ rounds, [0-9]+ accepted" logs/E-$name.log | awk '{r+=$2; a+=$4} END {print "COPY rounds", r, "accepted", a}'
    ./srv.sh stop
done
step "restoring the fast unit"
./deploy_fast.sh
step "N2E DONE"
