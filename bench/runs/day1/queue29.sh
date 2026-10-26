#!/bin/bash
# queue29: the 262K unit check, with the fast unit fully down first; then the fast unit back.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
down() { for i in $(seq 1 120); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || return 0; sleep 1; done; return 1; }
ready() { for i in $(seq 1 300); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null && return 0; sleep 1; done; return 1; }
systemctl stop gemma4-31b-fast; down
step "262K unit"
systemctl start gemma4-31b-262k && ready && systemctl is-active gemma4-31b-262k && \
    python3 verify.py http://127.0.0.1:8080 80000 && \
    python3 needle.py --label "U262 unit 200K" --lengths 200000 --depths 0.5
systemctl stop gemma4-31b-262k; down
step "fast unit back"
systemctl start gemma4-31b-fast && ready && systemctl is-active gemma4-31b-fast && curl -s http://127.0.0.1:8080/v1/models | head -c 120; echo
step "QUEUE29 DONE"
