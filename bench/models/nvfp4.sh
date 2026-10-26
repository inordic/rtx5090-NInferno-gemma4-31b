#!/bin/bash
R=melcheikh/gemma-4-31B-it-qat-NVFP4-Blackwell
D=/opt/models/dl2/qat-nvfp4
cd $D
for f in $(curl -s "https://huggingface.co/api/models/$R/tree/main" | python3 -c "import json,sys;[print(x[\"path\"]) for x in json.load(sys.stdin) if x[\"type\"]==\"file\"]"); do
  for i in 1 2 3; do curl -sL -C - --fail --retry 10 --retry-delay 10 -o "$f" "https://huggingface.co/$R/resolve/main/$f" && break; sleep 5; done
  echo "got $f $(stat -c%s $f)"
done
echo NVFP4_DL_DONE
