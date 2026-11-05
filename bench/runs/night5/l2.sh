#!/bin/bash
# l2: how often the known borderline 129K needle misses. The prompt differs only in its leading nonce.
# A: live unit, cache full of stress data. B: the same unit, freshly restarted. C: the previous setup, fresh.
source <(sed -n "/^cd \/root\/gbench/,/^counts() {/p" n5/night5.sh | sed "\$d")
OUT=n5/needle_rep.jsonl
trap './srv.sh stop >/dev/null 2>&1; ./deploy_fast.sh' EXIT
reps() {  # label
    for i in 1 2 3 4 5 6 7 8 9 10; do
        [ -e /root/gbench/HOLD ] && { echo "HOLD set"; exit 0; }
        curl -sf -m 10 http://127.0.0.1:8080/v1/models >/dev/null || { echo "server gone in $1"; exit 1; }
        python3 needle.py --label "$1" --lengths 140000 --depths 0.5 --out $OUT 2>&1 | grep '"L"' | cut -c1-140
    done
    echo "SUM $1: $(grep -c "\"label\": \"$1\", \"acc\": \"1/1\"" $OUT) of 10"
}
{
  step A; reps "L2 A live"
  step B; systemctl restart gemma4-31b-fast
  for i in $(seq 1 300); do curl -sf http://127.0.0.1:8080/v1/models >/dev/null && break; sleep 1; done
  reps "L2 B fresh"
  step C; systemctl stop gemma4-31b-fast
  for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models >/dev/null || break; sleep 1; done
  start L2_old $UP3 0 $(flags 202112 1024 $DR) && reps "L2 C old"
  ./srv.sh stop; ./deploy_fast.sh
  systemctl show gemma4-31b-fast -p NRestarts -p ActiveEnterTimestamp | tr "\n" " "; echo
  echo L2_DONE
} > n5/L2.txt 2>&1
