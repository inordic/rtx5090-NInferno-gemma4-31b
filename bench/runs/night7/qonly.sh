#!/bin/bash
# qonly.sh LABEL ARTIFACT SUITES: production flags, quality suites only (output n7/LABEL.txt)
L=$1; A=$2; cd /root/gbench
export NINFERNO_GEMMA4_SHARED_VALUES=2 NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_EMBED_HOST=1 NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64 NINFERNO_GEMMA4_SKIP_LAST_LAYER=1
PROD="--vision --max-context 203136 --prefill-chunk 1280 --kv-dtype fp8 --sliding-kv-dtype fp8 --spec mtp --drafter /opt/models/dl2/assistant.nvfp4.ninferno --draft-tokens 7 --chat-template /opt/models/templates/gemma4-agent.jinja --auto-tools-shared-prefix --auto-shared-prefix-requested"
{ systemctl stop gemma4-31b-fast; ./srv.sh stop > /dev/null 2>&1
  SRV_BIN=/opt/NInferno-up/build/apps-up4/ninferno-serve ./srv_any.sh start $L $A $PROD > /dev/null || { echo "$L DIED"; exit 1; }
  python3 geval.py --label "N7-$L" --suites $3 --conc 1 2>&1 | tail -4
  echo "=== $(date +%T) $L QONLY_DONE"; } > n7/$L.txt 2>&1
