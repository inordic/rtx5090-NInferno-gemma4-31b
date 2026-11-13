#!/bin/bash
# phase 3c (shorter refusal battery for C1 and L1): T1's KL (reference with BOS), the other candidates' batteries, T1's plain and MSE builds on the
# knowledge suites, then the stock reference and production against it
cd /root/gbench/n7
step() { echo "=== $(date +%T) $*"; }
R=/opt/models/n7/ref
until grep -qE "BATTERY_DONE|DIED" T1-imw.txt 2>/dev/null; do sleep 30; done; step "T1-imw battery: $(grep -E 'BATTERY_DONE|DIED' T1-imw.txt | tail -1)"
cd /root/gbench; python3 /opt/NInferno-up/tools/bench/hard_set.py --url http://127.0.0.1:8080/v1 --set /opt/NInferno-up/eval/corpora/hard50/hard50.json --tag n7_T1-imw --out hard50/n7_T1-imw.json > n7/T1-imw.hard50.txt 2>&1; cd n7; step "T1-imw hard50: $(tail -n 1 T1-imw.hard50.txt | cut -c1-120)"
/root/gbench/srv.sh stop > /dev/null 2>&1
./kld.sh ref $R/t1q5.ninferno /opt/models/n7/t1ref2.kld
for m in rtn mse imw; do ./kld.sh run /opt/models/n7/T1-$m.ninferno /opt/models/n7/t1ref2.kld T1-${m}_vs_t1; done
step "KL_T1_DONE"
./qonly.sh T1-rtn /opt/models/n7/T1-rtn.ninferno mmlupro,mmlu; step "T1-rtn: $(grep -E 'mmlu' T1-rtn.txt | head -2 | tr '\n' ' ')"
for c in C1 L1; do ./battery_full3.sh $c-rtn /opt/models/n7/$c.qkv-vision-nvfp4.ninferno; step "$(grep -E 'BATTERY_DONE|DIED' $c-rtn.txt | tail -1)"; done
for m in mse; do ./qonly.sh T1-$m /opt/models/n7/T1-$m.ninferno mmlupro,mmlu; step "T1-$m: $(grep -E 'mmlu' T1-$m.txt | head -2 | tr '\n' ' ')"; done
./kld.sh ref $R/stockq5.ninferno /opt/models/n7/stockref2.kld
./kld.sh run /opt/models/dl2/turbo-qkv-vision-nvfp4.r31.ninferno /opt/models/n7/stockref2.kld A0_vs_stock
./kld.sh run /opt/models/n7/T1-imw.ninferno /opt/models/n7/stockref2.kld T1-imw_vs_stock
step "PHASE3_DONE"
