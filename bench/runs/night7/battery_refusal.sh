#!/bin/bash
# refusal battery against the server on 8080: harmful (104), HarmBench (300), XSTest safe (250), OR-Bench-Hard sample (300); then the classifier
L=$1; cd /root/gbench/n7
for s in harmful harmbench xstest_safe orhard; do python3 -I refusaltest.py --label $L --set $s; done
nice -n 10 /opt/llama.cpp/.venv/bin/python -I judge.py refusal/$L.harmful.jsonl refusal/$L.harmbench.jsonl refusal/$L.xstest_safe.jsonl refusal/$L.orhard.jsonl 2>&1 | grep JUDGED
