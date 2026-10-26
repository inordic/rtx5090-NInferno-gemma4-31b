#!/bin/bash
# queue9: after queue8 - INT8-scores A/B (mode2 binary vs i8 binary): perplexity, top-1 agreement,
# prefill speed at 32K/128K, and a 128K needle through the i8 server.
cd /root/gbench
until grep -q "QUEUE8 DONE" queue8.log; do sleep 15; done
./srv.sh stop; sleep 2
QU=/opt/models/dl/.ninferno-cache/gemma-4-31B-it-qat-UD-Q4_K_XL.gemma4_published.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
step() { echo "=== $(date +%T) $*"; }
for v in mode2 i8; do
    G=/opt/NInferno/build/apps-$v/ninferno-gemma4
    for w in 4096 32768; do
        step "score $v window $w"
        $G $QU score --tokens /root/gbench/ppl64k.i32 --window $w --kv-dtype fp8 --context $w \
            --dump /root/gbench/score_${v}_$w.bin 2>&1 | tail -1
    done
    for n in 32768 131072; do
        step "bench $v prefill $n"
        $G $QU bench --tokens /root/gbench/prompt128k.i32 --prompt $n --decode 32 --kv-dtype fp8 \
            --context $((n + 256)) --chunk 1024 2>&1 | tail -1
    done
done
python3 /root/gbench/scorecmp.py
step "needle 128K through the i8 server"
cp -p /opt/NInferno/build/apps/ninferno-serve /root/gbench/ninferno-serve.mode2
sed -i "s#^S=.*#S=/opt/NInferno/build/apps-i8/ninferno-serve#" srv.sh
NINFERNO_GEMMA4_SHARED_VALUES=2 ./srv.sh start N1-i8-262k $QU --max-context 262144 --prefill-chunk 512 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 8 && \
    python3 needle.py --label "i8 shared2 262K" --lengths 128000 --depths 0.1,0.5,0.9 && \
    python3 gbench.py --label "N1 i8 QAT-UD K8 shared2 262K" --tg "" --skip-chat --pp 2048,8192,32768
./srv.sh stop
sed -i "s#^S=.*#S=/opt/NInferno/build/apps/ninferno-serve#" srv.sh
step "QUEUE9 DONE"
