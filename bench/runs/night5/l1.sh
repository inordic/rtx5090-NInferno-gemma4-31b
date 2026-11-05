#!/bin/bash
# l1: the live unit under agent-shaped load while traffic is away. No restart: every call goes to the running service.
cd /root/gbench
HS=/opt/NInferno-up/tools/bench/hard_set.py; SET=/opt/NInferno-up/eval/corpora/hard50/hard50.json
O=/root/gbench/n5
START=$(date -u "+%Y-%m-%d %H:%M:%S")
alive() { curl -sf -m 5 http://127.0.0.1:8080/v1/models >/dev/null; }
step() {
    n=$1; shift
    [ -e /root/gbench/HOLD ] && { echo "HOLD set, stopping"; return 1; }
    alive || { echo "server gone before $n"; return 1; }
    echo "--- $n $(date -u +%H:%M:%S)"
    timeout 1800 "$@" 2>&1 | tail -${TAIL:-4} | cut -c1-200
}
{
  echo "START $START"
  step replay1 python3 cachereplay.py --label "Live up4 r1" --slots 12   && step replay2 python3 cachereplay.py --label "Live up4 r2" --slots 12   && step replay3 python3 cachereplay.py --label "Live up4 r3" --slots 12   && step stress_agent python3 stress_entitlement.py --mode agent --rounds 300 --seed 8   && step stress_branch python3 stress_entitlement.py --mode branch --rounds 60 --seed 8   && TAIL=6 step needle python3 needle.py --label "Live up4 needle" --lengths 140000,170000,205000 --depths 0.5   && step hard50 python3 $HS --url http://127.0.0.1:8080/v1 --set $SET --tag live_up4 --out hard50/live_up4.json
  echo "--- totals $(date -u +%H:%M:%S)"
  J() { journalctl -u gemma4-31b-fast -o cat --since "$START"; }
  echo "requests done $(J | grep -c ' done | ') | failed $(J | grep -cE 'req#[0-9]+ failed|request failure') | exits $(J | grep -cE 'Engine failed|FATAL') | privately $(J | grep -c 'publishes privately')"
  systemctl show gemma4-31b-fast -p NRestarts -p ActiveEnterTimestamp | tr "\n" " "; echo
  echo "rss $(ps -o rss= -p $(systemctl show gemma4-31b-fast -p MainPID --value)) | gpu $(nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader)"
  J | grep -oE "mtp accepted [0-9]+/[0-9]+" | tr "/" " " | awk '{x+=$3;y+=$4} END {if (y) printf "mtp %d/%d (%.1f%%)\n",x,y,100*x/y}'
  echo L1_DONE
} > $O/L1.txt 2>&1
