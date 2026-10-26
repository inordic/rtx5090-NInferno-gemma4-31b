#!/bin/bash
# GPU queue 1: mode-2 262K benches, quality matrix, abl+MTP tune. Log: /root/gbench/queue1.log
cd /root/gbench
QU=/opt/models/dl/.ninferno-cache/gemma-4-31B-it-qat-UD-Q4_K_XL.gemma4_published.ninferno
QA=/opt/models/.ninferno-cache/gemma-4-31B_q4_0-it.gemma4_published.ninferno
AK=/opt/models/.ninferno-cache/gemma-4-31B-abliterated-Q4_K.gemma4_published.ninferno
A0=/opt/models/dl/.ninferno-cache/gemma-4-31B-abliterated-Q4_0.gemma4_published.ninferno
DR=/root/.cache/huggingface/hub/models--google--gemma-4-31B-it-assistant/ninferno/627c5ec1458b9086b841a91e0512fd31fd2fbbf1/gemma-4-31B-it-assistant.gemma4_assistant_published.q8_sl98304.ninferno
step() { echo "=== $(date +%T) $*"; }
step "C3 mode2 plain 262K chunk1024"
NINFERNO_GEMMA4_SHARED_VALUES=2 ./srv.sh start C3-qatud-plain-s2-262k $QU --max-context 262144 --prefill-chunk 1024 --kv-dtype fp8 --spec none && \
  python3 gbench.py --label "C3 QAT-UD plain shared2 262K" --skip-chat && python3 needle.py --label "C3 shared2 262K" --lengths 250000 --depths 0.1,0.5
./srv.sh stop
step "C4 mode2 MTP3 262K chunk512"
NINFERNO_GEMMA4_SHARED_VALUES=2 ./srv.sh start C4-qatud-mtp3-s2-262k $QU --max-context 262144 --prefill-chunk 512 --kv-dtype fp8 --spec mtp --drafter $DR --draft-tokens 3 && \
  python3 gbench.py --label "C4 QAT-UD MTP3 shared2 262K chunk512" && python3 needle.py --label "C4 shared2 MTP 262K" --lengths 250000 --depths 0.9
./srv.sh stop
step "abl Q4_0 + MTP tune"
./tune.sh $A0 q8 mtp:2,mtp:3,mtp:4 ablq40_q8
step "quality matrix"
./qmatrix.sh qatud-fp8 $QU 0 fp8
./qmatrix.sh qatud-shared1 $QU 1 fp8
./qmatrix.sh qatud-shared2 $QU 2 fp8
./qmatrix.sh qatud-bf16 $QU 0 bf16
./qmatrix.sh qat-q40-fp8 $QA 0 fp8
./qmatrix.sh abl-q4k-fp8 $AK 0 fp8
./qmatrix.sh abl-q40-fp8 $A0 0 fp8
step "QUEUE1 DONE"
