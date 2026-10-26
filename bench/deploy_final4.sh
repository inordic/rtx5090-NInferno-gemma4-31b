#!/bin/bash
# deploy_final4.sh: both model units onto apps-final4 (NInferno 6ff9931: request sums on /metrics). Unit files are
# backed up as *.bak-<date>; the fast unit restarts at a moment with no request in flight (waits up to 60 s).
set -e
d=$(date +%Y%m%d-%H%M%S)
for u in gemma4-31b-fast gemma4-31b-262k; do
    f=/etc/systemd/system/$u.service
    cp -p $f $f.bak-$d
    sed -i "s#/opt/NInferno/build/apps-final3/ninferno-serve#/opt/NInferno/build/apps-final4/ninferno-serve#" $f
    grep -o "apps-final[0-9]*" $f | head -1
done
systemctl daemon-reload
for i in $(seq 1 120); do
    a=$(curl -s -m 2 http://127.0.0.1:8080/health | python3 -c "import json,sys; print(json.load(sys.stdin).get(\"active_requests\", 1))" 2>/dev/null || echo 1)
    [ "$a" = 0 ] && break
    sleep 0.5
done
echo "in flight at restart: $a"
t0=$(date +%s.%N)
/root/gbench/deploy_fast.sh
echo "down for $(python3 -c "print(round($(date +%s.%N) - $t0, 1))") s"
