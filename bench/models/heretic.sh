#!/bin/bash
# Download llmfan46 QAT Heretic bf16, convert directly to QAT-aware Q4_0, remove bf16
R=llmfan46/gemma-4-31B-it-qat-q4_0-unquantized-uncensored-heretic
D=/opt/models/dl2/heretic-bf16
cd $D
for f in $(curl -s "https://huggingface.co/api/models/$R/tree/main" | python3 -c "import json,sys;[print(x[\"path\"]) for x in json.load(sys.stdin) if x[\"type\"]==\"file\"]"); do
  curl -sL -C - --fail --retry 10 --retry-delay 10 -o "$f" "https://huggingface.co/$R/resolve/main/$f" && echo "got $f"
done
ls -la $D
cd /opt/llama.cpp && GGUF_Q4_0_QAT=1 nice -n 10 .venv/bin/python convert_hf_to_gguf.py $D --outtype q4_0 --outfile /opt/models/dl2/heretic-qat-Q4_0-qataware.gguf 2>&1 | tail -3
ls -la /opt/models/dl2/heretic-qat-Q4_0-qataware.gguf && rm -rf $D && echo HERETIC_DONE
