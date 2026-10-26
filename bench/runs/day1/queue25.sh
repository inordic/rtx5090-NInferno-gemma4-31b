#!/bin/bash
# queue25: final binary (apps-final) - Q4 262K decode with and without the GeGLU split at every width.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
./srv.sh stop; sleep 2
QU=/opt/models/dl/.ninferno-cache/gemma-4-31B-it-qat-UD-Q4_K_XL.gemma4_published.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
sed "s#^S=.*#S=/opt/NInferno/build/apps-final/ninferno-serve#" srv.sh > srv_final.sh; chmod +x srv_final.sh
for m in 0 1; do
    step "Q$m: Q4 QAT-UD 262K chunk512, GeGLU split min T $m (final)"
    NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=$m NINFERNO_GEMMA4_SHARED_VALUES=2 ./srv_final.sh start Q$m-262k $QU --max-context 262144 --prefill-chunk 512 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 8 && \
        python3 gbench.py --label "Q$m QAT-UD final split$m K8 shared2 262K chunk512" --pp 2048,8192,32768,65536 --reps 1
    ./srv.sh stop
done
step "QUEUE25 DONE"
