#!/bin/bash
# qmatrix.sh <name> <artifact> <shared:0|1|2> <kv> [extra serve args]
cd /root/gbench
DR=/opt/models/dl2/qat-assistant.q40qat.ninferno
name=$1; art=$2; sh=$3; kv=$4; shift 4
NINFERNO_GEMMA4_SHARED_VALUES=$sh ./srv.sh start "q-$name" $art --max-context 4096 --max-concurrency 6 --kv-dtype $kv --spec mtp --drafter $DR --draft-tokens 3 "$@" || exit 1
python3 geval.py --label "$name" --suites gsm8k,mmlupro,humaneval --conc 6
./srv.sh stop
