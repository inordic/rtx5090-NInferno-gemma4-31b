#!/bin/bash
# watchdog10.sh: run once by a timer after night 10's window. If production is not running, stop night 10's jobs
# and start production unchanged.
cd /root/gbench/n10
if systemctl is-active -q gemma4-31b-fast; then echo "$(date -u +%T) production active, nothing to do"; exit 0; fi
echo "$(date -u +%T) production not active: stopping night 10's jobs"
[ -f gpu10.pid ] && { pkill -TERM -P "$(cat gpu10.pid)"; kill "$(cat gpu10.pid)" 2>/dev/null; }
pkill -f "[f]wd10.py|[t]rain10.py|[h]eldout10.py|[c]alib10.py|[n]inferno-perplexity|[h]ard_set.py|[g]en10.py"
sleep 3; ./srv10.sh stop; systemctl start gemma4-31b-fast
for i in $(seq 1 90); do curl -sf -m 5 http://127.0.0.1:8080/v1/models > /dev/null && break; sleep 5; done
echo "$(date -u +%T) production $(systemctl is-active gemma4-31b-fast)"
