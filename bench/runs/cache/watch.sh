#!/bin/bash
# overnight guard until 05:00 UTC: roll back once if the unit restarts, exits, fails requests or memory runs low
cd /root/gbench; U=/etc/systemd/system/gemma4-31b-fast.service; B=$(cat n6/backup_path); T0=$(cat n6/deploy_time)
END=$(date -u -d "tomorrow 05:00" +%s); [ $(date -u +%H) -lt 5 ] && END=$(date -u -d "today 05:00" +%s)
PID0=$(systemctl show gemma4-31b-fast -p MainPID --value)
while [ $(date -u +%s) -lt $END ]; do
    pid=$(systemctl show gemma4-31b-fast -p MainPID --value); act=$(systemctl is-active gemma4-31b-fast)
    mem=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    bad=$(journalctl -u gemma4-31b-fast -o cat --since "$T0" | grep -cE "Engine failed|FATAL|req#[0-9]+ failed|request failure")
    echo "$(date -u +%H:%M) active=$act pid=$pid mem_avail_mib=$mem bad=$bad" > n6/watch_status
    if [ "$act" != "active" ] || [ "$pid" != "$PID0" ] || [ "$mem" -lt 3000 ] || [ "$bad" -gt 0 ]; then
        echo "$(date -u +%H:%M:%S) ROLLBACK active=$act pid=$pid/$PID0 mem=$mem bad=$bad" >> n6/watch.log
        cp $B $U; systemctl daemon-reload; systemctl restart gemma4-31b-fast
        for i in $(seq 1 300); do curl -sf -m 5 http://127.0.0.1:8080/v1/models >/dev/null && break; sleep 1; done
        echo "$(date -u +%H:%M:%S) rolled back, active=$(systemctl is-active gemma4-31b-fast)" >> n6/watch.log; exit 1
    fi
    sleep 120
done
echo "$(date -u +%H:%M:%S) watch ended, no rollback" >> n6/watch.log
