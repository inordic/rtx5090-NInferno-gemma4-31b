#!/bin/bash
# the live unit after the deploy, for the README tables: every speed figure and the edit bench
cd /root/gbench
./idle_gate_any.sh 3
python3 gbench.py --label "Live up4" --pp 2048,8192,32768,65536,131072 --tg 2048,8192 --reps 2 2>&1 | tail -14 | cut -c1-200
python3 editbench.py --label "Live up4" 2>&1 | tail -3 | cut -c1-200
echo F5_DONE
