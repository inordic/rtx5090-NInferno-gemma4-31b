#!/bin/bash
# idle_gate.sh [minutes]: returns once the fast unit has served no request for that many minutes
# (default 45) and /root/gbench/HOLD does not exist. Overnight GPU tests start behind it.
M=${1:-45}
while :; do
    if [ ! -e /root/gbench/HOLD ] && systemctl is-active -q gemma4-31b-fast; then
        n=$(journalctl -u gemma4-31b-fast --since "-$M min" --no-pager -o cat 2>/dev/null | grep -c "req#")
        if [ "$n" = 0 ]; then exit 0; fi
    fi
    sleep 60
done
