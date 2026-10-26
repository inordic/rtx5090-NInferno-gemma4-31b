#!/bin/bash
# queue9b: after queue7d - INT8-scores attention A/B on the NVFP4 + GeGLU-split build:
# perplexity and top-1 agreement (lab tool), server prefill at 2K..64K, needle at 64K/128K.
cd /root/gbench
until grep -q "QUEUE7 DONE" queue7.log && grep -q BUILD_I8B_OK build_i8b.out; do sleep 15; done
./srv.sh stop; sleep 2
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=64
step() { echo "=== $(date +%T) $*"; }
for v in split i8b; do
    G=/opt/NInferno/build/apps-$v/ninferno-gemma4
    for w in 4096 32768; do
        step "score $v window $w"
        $G $QN score --tokens /root/gbench/ppl64k.i32 --window $w --kv-dtype fp8 --context $w \
            --dump /root/gbench/score_${v}_$w.bin 2>&1 | tail -1
    done
done
sed -e "s/\"mode2\", \"i8\"/\"split\", \"i8b\"/; s/out\[\"mode2\"\], out\[\"i8\"\]/out[\"split\"], out[\"i8b\"]/; s/ppl\['mode2'\]/ppl['split']/; s/ppl\['i8'\]/ppl['i8b']/; s/mode2 {/split {/; s/ i8 {/ i8b {/" scorecmp.py > scorecmp_b.py
python3 scorecmp_b.py
step "F7 serve i8b: NVFP4 + GeGLU split + INT8 scores, QAT drafter K8, shared2 140K chunk2048"
sed "s#^S=.*#S=/opt/NInferno/build/apps-i8b/ninferno-serve#" srv.sh > srv_i8b.sh; chmod +x srv_i8b.sh
NINFERNO_GEMMA4_SHARED_VALUES=2 ./srv_i8b.sh start F7-nvfp4-i8b $QN --max-context 140000 --prefill-chunk 2048 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 8 && \
    python3 gbench.py --label "F7 QAT-NVFP4 GeGLU-split INT8-scores K8 shared2 140K" --pp 2048,8192,32768,65536 --reps 1 && \
    python3 needle.py --label "F7 i8b needle" --lengths 64000,128000 --depths 0.1,0.5,0.9
./srv.sh stop
step "F8 serve split (no INT8): same config, needle + long prefill for reference"
NINFERNO_GEMMA4_SHARED_VALUES=2 ./srv.sh start F8-nvfp4-split-140k $QN --max-context 140000 --prefill-chunk 2048 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 8 && \
    python3 needle.py --label "F8 split needle" --lengths 64000,128000 --depths 0.1,0.5,0.9
./srv.sh stop
step "QUEUE9 DONE"
