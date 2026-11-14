#!/bin/bash
# chain10.sh: when the first batch is done, collect the second batch (jobs2) until one minute before the GPU phase
cd /root/gbench/n10
STOP_TS=$(( $(date -u -d "2026-11-14 01:30" +%s) - 60 ))
until grep -q GEN_DONE gen.out; do sleep 20; [ "$(date -u +%s)" -ge "$STOP_TS" ] && exit 0; done
echo "$(date -u +%T) batch 2 starts"
GEN_JOBS=/opt/models/n10/jobs2.jsonl timeout $(( STOP_TS - $(date -u +%s) )) /opt/venv-cu/bin/python -I gen10b.py run
echo "$(date -u +%T) batch 2 ended ($?)"
