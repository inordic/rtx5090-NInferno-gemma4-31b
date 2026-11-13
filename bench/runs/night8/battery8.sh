#!/bin/bash
# battery8.sh LABEL ARTIFACT: production's binary and flags (host cache at its default) with ARTIFACT on 8080;
# checks, speed, quality, hard50, style chats, the companion-agent set (companionfit.py). Output n8/LABEL.txt; the server stays up.
L=$1; A=$2; cd /root/gbench
export NINFERNO_GEMMA4_SHARED_VALUES=2 NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_EMBED_HOST=1 NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64 NINFERNO_GEMMA4_SKIP_LAST_LAYER=1
PROD="--vision --max-context 203136 --prefill-chunk 1280 --kv-dtype fp8 --sliding-kv-dtype fp8 --spec mtp --drafter /opt/models/dl2/assistant.nvfp4.ninferno --draft-tokens 7 --chat-template /opt/models/templates/gemma4-agent.jinja --auto-tools-shared-prefix --auto-shared-prefix-requested"
HS=/opt/NInferno-up/tools/bench/hard_set.py; SET=/opt/NInferno-up/eval/corpora/hard50/hard50.json
step() { echo "=== $(date +%T) $*"; }
{
systemctl stop gemma4-31b-fast; ./srv.sh stop > /dev/null 2>&1
step "$L: serving $A"
SRV_BIN=/opt/NInferno-up/build/apps-up4/ninferno-serve ./srv_any.sh start $L $A $PROD || { step "$L DIED"; exit 1; }
step "checks";  python3 verify.py 2>&1 | tail -3; python3 textcalltest.py 2>&1 | tail -1; python3 visiontest.py --label "N8 $L" 2>&1 | tail -1
python3 agenttest.py --label "N8 $L" --runs 1 2>&1 | tail -1
step "speed";   python3 gbench.py --label "N8 $L" --pp 2048,8192,32768 --reps 2 2>&1 | grep -E "^(PP|TG|CHAT)"
step "quality"; python3 geval.py --label "N8-$L" --suites mmlupro,mmlu,gsm8k,humaneval --conc 1 2>&1 | grep -E "^(mmlupro|mmlu|gsm8k|humaneval):"
step "hard50";  python3 $HS --url http://127.0.0.1:8080/v1 --set $SET --tag n8_$L --out hard50/n8_$L.json > n8/$L.hard50.txt 2>&1; tail -n 1 n8/$L.hard50.txt | cut -c1-140
python3 -c "
import json; r=json.load(open('hard50/n8_$L.json'))['rows']; a=sum(x['accepted'] for x in r); d=sum(x['drafted'] for x in r)
print('HARD50 $L', sum(1 for x in r if x['correct']), '/', len(r), 'drafts accepted %.1f%%' % (100*a/d))"
step "style";   python3 n4/n4style.py --label N8-$L 2>&1 | tail -1
step "companionfit"; python3 -I n8/companionfit.py $L 2>&1 | tail -1
grep -hE "Engine failed|FATAL" logs/$L.log | head -3
step "$L BATTERY_DONE"
} > n8/$L.txt 2>&1
