#!/bin/bash
# queue27: after queue26 - install both units; 262K unit: verify + one 200K needle; then the fast
# unit: full bench, verify, needles at 32K / 72K / 140K; leave the fast unit serving 8080.
cd /root/gbench
until grep -qE "QUEUE26 DONE" queue26.log 2>/dev/null; do sleep 10; done
step() { echo "=== $(date +%T) $*"; }
./srv.sh stop; sleep 2
cp units/gemma4-31b-fast.service units/gemma4-31b-262k.service /etc/systemd/system/
systemctl daemon-reload
ready() { for i in $(seq 1 300); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null && return 0; sleep 1; done; return 1; }
step "262K unit"
systemctl start gemma4-31b-262k && ready && systemctl is-active gemma4-31b-262k && \
    python3 verify.py http://127.0.0.1:8080 80000 && \
    python3 needle.py --label "U262 unit 200K" --lengths 200000 --depths 0.5
systemctl stop gemma4-31b-262k; sleep 3
step "fast unit"
systemctl start gemma4-31b-fast && ready && systemctl is-active gemma4-31b-fast && \
    python3 gbench.py --label "UF fast unit final (qk8p1 split1 155K chunk1024)" --pp 2048,8192,32768,65536 --reps 1 && \
    python3 verify.py http://127.0.0.1:8080 80000 && \
    python3 needle.py --label "UF fast unit needles" --lengths 32000,72000,140000 --depths 0.1,0.5,0.9
systemctl is-active gemma4-31b-fast
step "QUEUE27 DONE"
