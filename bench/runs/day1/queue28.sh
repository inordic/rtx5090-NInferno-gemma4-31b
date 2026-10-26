#!/bin/bash
# queue28: after queue27 - the 262K unit check verify.py cut short (verify + a 200K needle), then
# the fast unit back on 8080.
cd /root/gbench
until grep -qE "QUEUE27 DONE" queue27.log 2>/dev/null; do sleep 10; done
step() { echo "=== $(date +%T) $*"; }
ready() { for i in $(seq 1 300); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null && return 0; sleep 1; done; return 1; }
step "262K unit (again)"
systemctl start gemma4-31b-262k && ready && systemctl is-active gemma4-31b-262k && \
    python3 verify.py http://127.0.0.1:8080 80000 && \
    python3 needle.py --label "U262 unit 200K" --lengths 200000 --depths 0.5
systemctl stop gemma4-31b-262k; sleep 3
step "fast unit back"
systemctl start gemma4-31b-fast && ready && systemctl is-active gemma4-31b-fast
step "QUEUE28 DONE"
