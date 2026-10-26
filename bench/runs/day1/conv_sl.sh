#!/bin/bash
# conv_sl.sh: QAT drafter variants with a shortlist head (Gemma 4 ranking gemma4_ranking2), from
# the QAT-aware Q4_0 GGUF; the layers stay QAT Q4_0, the restricted head is quantized as named.
cd /opt/NInferno
SRC=/opt/models/dl2/qat-assistant-Q4_0-qat.gguf
R=/opt/NInferno/tools/spec/rankings/gemma4_ranking2.i64
for v in "65536 q8" "98304 q8" "65536 q4"; do
    set -- $v
    out=/opt/models/dl2/qat-assistant.sl$(( $1 / 1024 ))k-$2.ninferno
    nice -n 10 build/apps/ninferno-convert --model $SRC --recipe gemma4_assistant_published \
        --proposal --proposal-rows $1 --ranking $R --drafter-quant $2 --out $out 2>&1 | grep -vE "^resource|^gguf" | tail -2
    ls -la $out 2>/dev/null && echo "CONV_$1_$2_OK"
done
echo CONV_DONE
