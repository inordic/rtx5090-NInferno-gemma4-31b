#!/bin/bash
# usage: srv.sh start <name> <model> [serve args...] | srv.sh stop
S=/opt/NInferno/build/apps-split/ninferno-serve
L=/root/gbench/logs; mkdir -p $L
if [ "$1" = stop ]; then
  pkill -INT -f "apps[^ ]*/ninferno-serve" ; for i in $(seq 1 60); do pgrep -f "apps[^ ]*/ninferno-serve" >/dev/null || break; sleep 1; done
  pkill -9 -f "apps[^ ]*/ninferno-serve" 2>/dev/null; sleep 2; exit 0
fi
name=$2; model=$3; shift 3
cd /opt/models
PATH=/usr/local/cuda-13.2/bin:$PATH nohup $S $model --host 0.0.0.0 --port 8080 --model-id "$name" "$@" > $L/$name.log 2>&1 &
for i in $(seq 1 600); do
  if grep -q "listening on" $L/$name.log; then break; fi
  if ! pgrep -f "apps[^ ]*/ninferno-serve" >/dev/null && sleep 2 && ! pgrep -f "apps[^ ]*/ninferno-serve" >/dev/null && ! grep -q "listening on" $L/$name.log; then echo "DIED"; grep -E "FATAL|error|refus" $L/$name.log | tail -5; exit 1; fi
  sleep 1
done
grep -E "plan \||capacity \||speculation \| serves|context \||engine ready|choice \|" $L/$name.log | sed -E "s/^.*(INFO|WARN) +//" | cut -c1-260
