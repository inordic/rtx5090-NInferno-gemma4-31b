#!/bin/bash
# queue21: after queue17 + build_qk8 - FP8 scores (E4M3 Q x K codes) + FP8 P V in the lab tool:
# 32K perplexity against split 76.96 / a8 78.56 / pv8b 77.99, prefill at 32K / 64K (pv8b 3,522 / 2,542).
cd /root/gbench
until grep -q "QUEUE17 DONE" queue17.log 2>/dev/null && grep -qE "BUILD_QK8_OK|BUILD_FAILED" build_qk8.out; do sleep 15; done
step() { echo "=== $(date +%T) $*"; }
grep -q BUILD_QK8_OK build_qk8.out || { step "QK8 BUILD FAILED"; exit 1; }
./srv.sh stop; sleep 2
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
G=/opt/NInferno/build/apps-qk8/ninferno-gemma4
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=64
step "score qk8 window 32768"
$G $QN score --tokens /root/gbench/ppl64k.i32 --window 32768 --kv-dtype fp8 --context 32768 --dump /root/gbench/score_qk8_32768.bin > /dev/null 2>&1
python3 ppl.py /root/gbench/score_qk8_32768.bin
for p in 8192 32768 65536; do
    step "bench qk8 prefill $p"
    $G $QN bench --tokens /root/gbench/prompt128k.i32 --prompt $p --decode 4 --kv-dtype fp8 --context $((p + 2048)) --chunk 2048 2>&1 | grep -oE "\"prefill_tok_s\":[0-9.]+|\"decode_tok_s\":[0-9.]+" | tr "\n" " "; echo
done
step "bench pv8b prefill 8192 (reference)"
/opt/NInferno/build/apps-pv8b/ninferno-gemma4 $QN bench --tokens /root/gbench/prompt128k.i32 --prompt 8192 --decode 4 --kv-dtype fp8 --context 10240 --chunk 2048 2>&1 | grep -oE "\"prefill_tok_s\":[0-9.]+" ; echo
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
QU=/opt/models/dl/.ninferno-cache/gemma-4-31B-it-qat-UD-Q4_K_XL.gemma4_published.ninferno
sed "s#^S=.*#S=/opt/NInferno/build/apps-qk8/ninferno-serve#" srv.sh > srv_qk8.sh; chmod +x srv_qk8.sh
rp() {  # the exact P7 miss (first draw, depth 0.5) on one build
    NINFERNO_GEMMA4_SHARED_VALUES=2 ./$2 start "R-$1" $3 --max-context 155000 --prefill-chunk 1024 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 8 && \
        python3 needle.py --label "R140 $1" --lengths 140000 --depths 0.5
    ./srv.sh stop
}
step "repro P7 miss: i8c"; rp nvfp4-i8c srv_i8c.sh $QN
step "repro P7 miss: split"; rp nvfp4-split srv.sh $QN
step "repro P7 miss: qk8"; rp nvfp4-qk8 srv_qk8.sh $QN
step "repro P7 miss: Q4 QAT-UD"; rp qatud srv.sh $QU
step "QUEUE21 DONE"
