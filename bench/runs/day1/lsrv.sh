#!/bin/bash
# lsrv.sh start <name> <bin-dir> [llama-server args...] | lsrv.sh stop
L=/root/gbench/logs; mkdir -p $L
if [ "$1" = stop ]; then pkill -INT -f "llama-server" ; for i in $(seq 1 30); do pgrep -f llama-server >/dev/null || break; sleep 1; done; pkill -9 -f llama-server 2>/dev/null; sleep 2; exit 0; fi
name=$2; bin=$3; shift 3
nohup $bin/llama-server --host 0.0.0.0 --port 8080 -ngl 999 -fa on --jinja "$@" > $L/$name.log 2>&1 &
for i in $(seq 1 600); do
  if grep -qE "server is listening|HTTP server listening" $L/$name.log; then break; fi
  if ! pgrep -f llama-server >/dev/null; then echo DIED; tail -5 $L/$name.log; exit 1; fi
  sleep 1
done
grep -E "CUDA0 (model|KV)|draft|spec|n_ctx|offloaded" $L/$name.log | head -12 | cut -c1-200
