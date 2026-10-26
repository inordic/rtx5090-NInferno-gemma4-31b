#!/bin/bash
# Download melcheikh/gemma-4-31B-it-qat-NVFP4-mse-Blackwell and convert it like r31 (Q8 table, Q6 head).
R=melcheikh/gemma-4-31B-it-qat-NVFP4-mse-Blackwell
D=/opt/models/dl2/qat-nvfp4-mse
cd $D
for f in $(curl -s "https://huggingface.co/api/models/$R/tree/main" | python3 -c "import json,sys;[print(x[\"path\"]) for x in json.load(sys.stdin) if x[\"type\"]==\"file\"]"); do
  for i in 1 2 3 4 5; do curl -sL -C - --fail --retry 10 --retry-delay 10 -o "$f" "https://huggingface.co/$R/resolve/main/$f" && break; sleep 5; done
  echo "got $f $(stat -c%s $f)"
done
echo MSE_DL_DONE
cd /opt/models/dl2 && nice -n 15 /opt/NInferno/build/apps/ninferno-convert --model $D --recipe gemma4_31b_nvfp4 --out /opt/models/dl2/qat-nvfp4-mse.r31.ninferno --components text 2>&1 | tail -4
ls -la /opt/models/dl2/qat-nvfp4-mse.r31.ninferno && echo MSE_CONVERT_DONE
