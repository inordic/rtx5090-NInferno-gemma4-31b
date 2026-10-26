#!/bin/bash
# prep_turbo_vision: LilaRest NVFP4-turbo again plus the vision tower of Google's original (non-QAT)
# Gemma 4 31B IT (unsloth/gemma-4-31B-it-GGUF mmproj-BF16; the QAT repo's projector differs), converted
# with Q/K/V packed (NINFERNO_CONVERT_NORMED_QKV_DIVISOR=1) into one text+vision artifact. No GPU.
step() { echo "=== $(date +%T) $*"; }
get() {  # get <repo> <dir> [file]
    mkdir -p $2 && cd $2
    for f in $(curl -s "https://huggingface.co/api/models/$1/tree/main" | python3 -c "import json,sys;[print(x[\"path\"]) for x in json.load(sys.stdin) if x[\"type\"]==\"file\"]"); do
        [ -n "$3" ] && [ "$f" != "$3" ] && continue
        nice -n 19 ionice -c3 curl -sL -C - --fail --retry 10 --retry-delay 10 -o "$f" "https://huggingface.co/$1/resolve/main/$f" && echo "got $f $(stat -c%s $f)"
    done
}
step "Google original IT vision projector (mmproj-BF16)"; get unsloth/gemma-4-31B-it-GGUF /opt/models/dl3/it-mmproj mmproj-BF16.gguf
step "LilaRest NVFP4-turbo source"; get LilaRest/gemma-4-31B-it-NVFP4-turbo /opt/models/dl3/turbo-src
du -sh /opt/models/dl3/turbo-src /opt/models/dl3/it-mmproj
python3 -c "
import json
c = json.load(open(\"/opt/models/dl3/turbo-src/config.json\"))
print(\"vision_config in config:\", \"vision_config\" in c)
i = json.load(open(\"/opt/models/dl3/turbo-src/model.safetensors.index.json\"))[\"weight_map\"]
print(\"vision tensors in checkpoint:\", sum(1 for k in i if \"vision\" in k or \"multi_modal\" in k or \"embed_vision\" in k))
"
step "GET DONE"
