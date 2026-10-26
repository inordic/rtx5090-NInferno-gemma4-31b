#!/bin/bash
# watch_n2.sh: wait until the night-2 logs gain a milestone (a batch done) or a failure, print the new lines.
cd /root/gbench
pat="N2[A-J] DONE|DIED|Traceback|Error|FAILED|artifact missing|Killed|out of memory"
count() { cat n2?_*.log 2>/dev/null | grep -cE "$pat"; }
c0=$(count)
for i in $(seq 1 1440); do
    sleep 15
    if [ "$(count)" != "$c0" ]; then sleep 5; for f in n2?_*.log; do grep -HnE "$pat" $f; done | tail -n +$((c0 + 1)); exit 0; fi
done
echo "watch timeout"
