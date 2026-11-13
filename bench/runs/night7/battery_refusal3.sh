#!/bin/bash
# shorter refusal battery: harmful (104) and the first 100 HarmBench prompts at 384 tokens, XSTest safe (250) and
# the first 150 OR-Bench-Hard prompts at 128 tokens; then the classifier
L=$1; cd /root/gbench/n7
python3 -I refusaltest.py --label $L --set harmful
python3 -I refusaltest.py --label $L --set harmbench --limit 100
python3 -I refusaltest.py --label $L --set xstest_safe --max-tokens 128
python3 -I refusaltest.py --label $L --set orhard --limit 150 --max-tokens 128
nice -n 10 /opt/llama.cpp/.venv/bin/python -I judge.py refusal/$L.harmful.jsonl refusal/$L.harmbench.jsonl refusal/$L.xstest_safe.jsonl refusal/$L.orhard.jsonl 2>&1 | grep JUDGED
