#!/bin/bash
# idle_gate_any.sh [minutes]: like idle_gate.sh, but passes while the fast unit is down too (a test
# batch left it stopped); it checks the unit's journal for requests either way.
M=${1:-45}
while :; do
    if [ ! -e /root/gbench/HOLD ]; then
        n=$(journalctl -u gemma4-31b-fast --since "-$M min" --no-pager -o cat 2>/dev/null | grep -c "req#")
        if [ "$n" = 0 ]; then exit 0; fi
    fi
    sleep 60
done
