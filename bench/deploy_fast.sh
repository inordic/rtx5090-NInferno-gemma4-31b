#!/bin/bash
# Put the fast unit on 8080: stop anything serving there, start gemma4-31b-fast, wait until it answers.
systemctl stop gemma4-31b-262k
/root/gbench/srv.sh stop > /dev/null 2>&1
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
systemctl start gemma4-31b-fast
for i in $(seq 1 300); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null && break; sleep 1; done
systemctl is-active gemma4-31b-fast
