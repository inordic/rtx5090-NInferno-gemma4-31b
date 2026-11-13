#!/bin/bash
# candidates' refusal battery: harmful (104) and HarmBench (300) at 384 tokens, XSTest safe (250) and the
# first 150 OR-Bench-Hard prompts at 256 tokens; then the classifier
L=$1; cd /root/gbench/n7
python3 -I refusaltest.py --label $L --set harmful
python3 -I refusaltest.py --label $L --set harmbench
python3 -I refusaltest.py --label $L --set xstest_safe --max-tokens 256
python3 -I refusaltest.py --label $L --set orhard --limit 150 --max-tokens 256
nice -n 10 /opt/llama.cpp/.venv/bin/python -I judge.py refusal/$L.harmful.jsonl refusal/$L.harmbench.jsonl refusal/$L.xstest_safe.jsonl refusal/$L.orhard.jsonl 2>&1 | grep JUDGED
