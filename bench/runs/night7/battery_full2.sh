#!/bin/bash
# battery_full.sh LABEL ARTIFACT [suites]: production's binary and flags (host cache at its default) on 8080 with
# ARTIFACT, then checks, speed, quality, hard50 and the refusal battery. Output n7/LABEL.txt; production stays stopped.
L=$1; A=$2; cd /root/gbench
export NINFERNO_GEMMA4_SHARED_VALUES=2 NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_EMBED_HOST=1 NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64 NINFERNO_GEMMA4_SKIP_LAST_LAYER=1
PROD="--vision --max-context 203136 --prefill-chunk 1280 --kv-dtype fp8 --sliding-kv-dtype fp8 --spec mtp --drafter /opt/models/dl2/assistant.nvfp4.ninferno --draft-tokens 7 --chat-template /opt/models/templates/gemma4-agent.jinja --auto-tools-shared-prefix --auto-shared-prefix-requested"
HS=/opt/NInferno-up/tools/bench/hard_set.py; SET=/opt/NInferno-up/eval/corpora/hard50/hard50.json
step() { echo "=== $(date +%T) $*"; }
{
systemctl stop gemma4-31b-fast; ./srv.sh stop > /dev/null 2>&1
step "$L: serving $A"
SRV_BIN=/opt/NInferno-up/build/apps-up4/ninferno-serve ./srv_any.sh start $L $A $PROD || { step "$L DIED"; exit 1; }
step "checks";  python3 verify.py 2>&1 | tail -3; python3 textcalltest.py 2>&1 | tail -1; python3 visiontest.py --label "N7 $L" 2>&1 | tail -1
python3 agenttest.py --label "N7 $L" --runs 1 2>&1 | tail -1
step "speed";   python3 gbench.py --label "N7 $L" --pp 2048,8192,32768 --reps 2 2>&1 | tail -14
step "quality"; python3 geval.py --label "N7-$L" --suites ${3:-mmlupro,mmlu,gsm8k,humaneval} --conc 1 2>&1 | tail -6
step "hard50";  python3 $HS --url http://127.0.0.1:8080/v1 --set $SET --tag n7_$L --out hard50/n7_$L.json 2>&1 | tail -3
step "refusal"; n7/battery_refusal2.sh $L 2>&1
grep -oE "mtp accepted [0-9]+/[0-9]+" logs/$L.log | tr '/' ' ' | awk -v L=$L '{x += $3; y += $4} END {if (y) printf "MTP %s accepted %d/%d (%.1f%%)\n", L, x, y, 100 * x / y}'
grep -hE "Engine failed|FATAL" logs/$L.log | head -3
step "$L BATTERY_DONE"
} > n7/$L.txt 2>&1
