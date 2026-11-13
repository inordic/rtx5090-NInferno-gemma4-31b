#!/bin/bash
# measure9.sh LABEL DRAFTER: Artemis on production's binary and flags with DRAFTER, on 8080; speed, hard50 and the
# held-out prompts, with acceptance from the server log. Output n9/m-LABEL.txt; the server stays up.
L=$1; DR=$2; cd /root/gbench
A=/opt/models/artemis/artemis-v1.2-imw-qkv-vision-nvfp4.r31.ninferno
export NINFERNO_GEMMA4_SHARED_VALUES=2 NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_EMBED_HOST=1 NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64 NINFERNO_GEMMA4_SKIP_LAST_LAYER=1
PROD="--vision --max-context 203136 --prefill-chunk 1280 --kv-dtype fp8 --sliding-kv-dtype fp8 --spec mtp --drafter $DR --draft-tokens 7 --chat-template /opt/models/templates/gemma4-agent.jinja --auto-tools-shared-prefix --auto-shared-prefix-requested"
HS=/opt/NInferno-up/tools/bench/hard_set.py; SET=/opt/NInferno-up/eval/corpora/hard50/hard50.json
step() { echo "=== $(date +%T) $*"; }
{
systemctl stop gemma4-31b-fast; ./srv.sh stop > /dev/null 2>&1
step "$L: serving Artemis with $DR"
SRV_BIN=/opt/NInferno-up/build/apps-up4/ninferno-serve ./srv_any.sh start $L $A $PROD || { step "$L DIED"; exit 1; }
step "checks"; python3 verify.py 2>&1 | tail -3; python3 textcalltest.py 2>&1 | tail -1
step "speed";  python3 gbench.py --label "N9 $L" --pp 2048 --reps 2 2>&1 | grep -E "^(TG|CHAT)"
step "hard50"; python3 $HS --url http://127.0.0.1:8080/v1 --set $SET --tag n9_$L --out hard50/n9_$L.json > n9/$L.hard50.txt 2>&1; tail -n 1 n9/$L.hard50.txt | cut -c1-140
python3 -c "
import json; r=json.load(open('hard50/n9_$L.json'))['rows']; a=sum(x['accepted'] for x in r); d=sum(x['drafted'] for x in r)
print('HARD50 $L', sum(1 for x in r if x['correct']), '/', len(r), 'drafts accepted %.1f%%' % (100*a/d))"
step "heldout"; python3 n9/heldout.py $L logs/$L.log
grep -hE "Engine failed|FATAL" logs/$L.log | head -3
step "$L MEASURE_DONE"
} > n9/m-$L.txt 2>&1
