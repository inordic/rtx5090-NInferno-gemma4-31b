#!/bin/bash
# dl.sh REPO NAME: download a BF16 checkpoint into /opt/models/n8/NAME/bf16 and check the LFS SHA256s (no GPU)
R=$1; N=$2; B=/opt/models/n8/$N; mkdir -p $B/bf16; cd $B/bf16
step() { echo "=== $(date +%T) $*"; }
step "listing $R"
curl -s "https://huggingface.co/api/models/$R/tree/main" > ../tree.json
python3 -c "
import json; t=json.load(open('../tree.json'))
if isinstance(t, dict): print('ERR', t); raise SystemExit(1)
for x in t:
    if x['type']=='file' and not x['path'].endswith(('.gguf','.png','.jpg')) and '/' not in x['path']:
        print(x['path'], (x.get('lfs') or {}).get('oid',''))" > ../files.txt || { step "DL_FAILED (listing)"; exit 1; }
: > ../sha.txt
while read f sha; do
    for i in 1 2 3 4 5; do curl -sL -C - --fail --retry 10 --retry-delay 10 -o "$f" "https://huggingface.co/$R/resolve/main/$f" && break; sleep 15; done
    [ -n "$sha" ] && echo "$sha  $f" >> ../sha.txt
done < ../files.txt
du -sh .
sha256sum -c --quiet ../sha.txt && step "DL_DONE $N" || step "DL_FAILED (sha) $N"
