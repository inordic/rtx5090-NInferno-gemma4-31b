#!/bin/bash
# prep_night2b: after prep_night2 - the QAT-UD Q4 target with its vision projector (mmproj-BF16)
# named explicitly; the CLI converter does not pick the projector up by itself.
until grep -q "PREP DONE" /root/gbench/prep_night2.log; do sleep 30; done
echo "=== $(date +%T) convert QAT-UD Q4 + vision (mmproj-BF16)"
cd /opt/models/dl && nice -n 19 ionice -c3 /opt/NInferno/build/apps/ninferno-convert --model /opt/models/dl/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf \
    --source vision=/opt/models/dl/mmproj-BF16.gguf --components text,vision --out /opt/models/dl/qat-ud-q4-vision.ninferno 2>&1 | grep -vE "^(resource|gguf resources)" | tail -4
ls -la /opt/models/dl/qat-ud-q4-vision.ninferno && echo "=== $(date +%T) PREP2 DONE"
