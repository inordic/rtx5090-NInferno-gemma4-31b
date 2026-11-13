#!/bin/bash
# phase 4b: T1-imw's knowledge suites alone, C1 in the importance-weighted build (T1's imatrix as a proxy: both
# edit only o_proj/down_proj of the same base), then production back, its knowledge suites tonight, the model judge
cd /root/gbench/n7
step() { echo "=== $(date +%T) $*"; }
R=/opt/models/n7/ref
until grep -q PHASE3_DONE phase3d.out; do sleep 30; done
./qonly.sh T1-imw-q /opt/models/n7/T1-imw.ninferno mmlupro,mmlu; step "T1-imw again: $(grep -E '^mmlu' T1-imw-q.txt | tr '\n' ' ')"
/root/gbench/srv.sh stop > /dev/null 2>&1
/opt/venv-cu/bin/python hlquant2.py /opt/models/n7/C1/bf16 /opt/models/n7/C1/q /root/gbench/n4/lila_scales.json $R/gemma-4-31b-it-heretic-ara.imatrix.gguf --device=cuda --methods=imw > quant2_C1.log 2>&1
tail -n 1 quant2_C1.log
./conv.sh /opt/models/n7/C1/q-imw /opt/models/n7/C1/bf16 /opt/models/n7/C1-imw.ninferno | tail -1; rm -rf /opt/models/n7/C1/q-imw
./qonly.sh C1-imw /opt/models/n7/C1-imw.ninferno mmlupro,mmlu,gsm8k,humaneval; step "C1-imw: $(grep -E '^(mmlu|gsm|human)' C1-imw.txt | tr '\n' ' ')"
cd /root/gbench; python3 /opt/NInferno-up/tools/bench/hard_set.py --url http://127.0.0.1:8080/v1 --set /opt/NInferno-up/eval/corpora/hard50/hard50.json --tag n7_C1-imw --out hard50/n7_C1-imw.json > n7/C1-imw.hard50.txt 2>&1; cd n7
step "C1-imw hard50: $(tail -n 1 C1-imw.hard50.txt | cut -c1-110)"
step "production back"; /root/gbench/deploy_fast.sh
for i in $(seq 1 60); do curl -sf -m 5 http://127.0.0.1:8080/v1/models > /dev/null && break; sleep 5; done
cd /root/gbench; python3 geval.py --label "N7-A0-q" --suites mmlupro,mmlu --conc 1 2>&1 | tail -3 > n7/A0-q.txt; cd n7; step "production tonight: $(grep -E '^mmlu' A0-q.txt | tr '\n' ' ')"
step "model judge"
for L in A0 T1-imw C1-rtn L1-rtn; do python3 -I judge_llm.py refusal/$L.harmful.jsonl refusal/$L.harmbench.jsonl refusal/$L.xstest_safe.jsonl refusal/$L.orhard.jsonl; done
step "PHASE4_DONE"
