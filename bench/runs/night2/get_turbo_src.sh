#!/bin/bash
# get_turbo_src: download LilaRest NVFP4-turbo again (no GPU) for the fused-QKV re-conversion.
step() { echo "=== $(date +%T) $*"; }
get() {  # get <repo> <dir>
    mkdir -p $2 && cd $2
    for f in $(curl -s "https://huggingface.co/api/models/$1/tree/main" | python3 -c "import json,sys;[print(x[\"path\"]) for x in json.load(sys.stdin) if x[\"type\"]==\"file\"]"); do
        nice -n 19 ionice -c3 curl -sL -C - --fail --retry 10 --retry-delay 10 -o "$f" "https://huggingface.co/$1/resolve/main/$f" && echo "got $f $(stat -c%s $f)"
    done
}
step "LilaRest NVFP4-turbo source"; get LilaRest/gemma-4-31B-it-NVFP4-turbo /opt/models/dl3/turbo-src
du -sh /opt/models/dl3/turbo-src
step "GET DONE"
