#!/bin/bash
# gpu9.sh: the GPU phase of the drafter retraining. Waits until enough replies are collected (N_STOP) or the clock
# reaches T_STOP (UTC HHMM), stops collection, then with production stopped: Artemis's imw weights, the
# layer-by-layer pass, drafter training, conversion, old-vs-new measurement. Any failure brings production back at once.
exec 9> /root/gbench/n9/gpu9.lock; flock -n 9 || { echo "gpu9 already running"; exit 1; }
cd /root/gbench/n9
N_STOP=${N_STOP:-2400}; T_STOP=${T_STOP:-1700}
PY=/opt/venv-cu/bin/python
export NINFERNO_GEMMA4_SHARED_VALUES=2 NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_EMBED_HOST=1 NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64 NINFERNO_GEMMA4_SKIP_LAST_LAYER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo "=== $(date -u +%T) $*"; }
back() {
    step "production back ($1)"; /root/gbench/srv.sh stop > /dev/null 2>&1; systemctl start gemma4-31b-fast
    for i in $(seq 1 60); do curl -sf -m 5 http://127.0.0.1:8080/v1/models > /dev/null && break; sleep 5; done
    step "GPU9_END: $1; production $(systemctl is-active gemma4-31b-fast)"; exit 0
}
until [ "$(wc -l < /opt/models/n9/gen.jsonl)" -ge "$N_STOP" ] || [ "$(date -u +%H%M)" -ge "$T_STOP" ] || grep -q GEN_DONE gen.out; do sleep 60; done
pkill -INT -f "[g]en.py run"; sleep 10
step "collected $(wc -l < /opt/models/n9/gen.jsonl) replies"
$PY -I fwd.py tokenize 2>&1 | grep TOKENIZED
for i in $(seq 1 240); do
    a=$(curl -s -m 5 http://127.0.0.1:8080/health | python3 -c "import json,sys; print(json.load(sys.stdin)['active_requests'])" 2>/dev/null)
    [ "$a" = "0" ] && break; sleep 5
done
step "stopping production"; systemctl stop gemma4-31b-fast; /root/gbench/srv.sh stop > /dev/null 2>&1; sleep 3
step "imw weights"
(cd /root/gbench/n7 && $PY hlquant2.py /opt/models/n8/A2/bf16 /opt/models/n9/A2-q /root/gbench/n4/lila_scales.json /opt/models/n8/ref/A2.imatrix.gguf --device=cuda --methods=imw) > quant.log 2>&1 || back "quantizer failed"
grep -E "imatrix missing|DONE" quant.log | tail -2
step "forward pass"
$PY -I fwd.py run imw cuda > fwd.log 2>&1 || { tail -5 fwd.log; back "forward pass failed"; }
grep -E "pass .* written|FWD_DONE" fwd.log | tail -2
step "training"
ACCUM=4096 $PY -I train.py train /opt/models/n9/states-imw /opt/models/n9/asst 3 5e-5 > train.log 2>&1 || { tail -5 train.log; back "training failed"; }
grep -E "base drafter|train:|epoch [0-9]+:" train.log | cut -c1-300
BEST=$(python3 -c "
import json,re
best=None
for l in open('train.log'):
    m=re.match(r'.*epoch (\d+): (\{.*\})', l)
    if m:
        a=json.loads(m.group(2))['accepted_per_round']
        if best is None or a>best[1]: best=(m.group(1),a)
print(best[0])")
step "converting epoch $BEST"
/opt/NInferno-up/build/apps/ninferno-convert --model /opt/models/n9/asst-ep$BEST --recipe gemma4_assistant_published --drafter-quant nvfp4 \
    --out /opt/models/n9/assistant-artemis-ep$BEST.nvfp4.ninferno > convert.log 2>&1 || back "conversion failed"
tail -1 convert.log | cut -c1-160
step "resting the GPU 5 min"; sleep 300
./measure9.sh old /opt/models/dl2/assistant.nvfp4.ninferno; grep -E "^(TG|CHAT|HARD50|HELDOUT .* (all|greedy|sampled):)|DIED" m-old.txt
step "needle 140K on a rested GPU (prefill check)"; /root/gbench/srv.sh stop > /dev/null 2>&1; sleep 300
SRV_BIN=/opt/NInferno-up/build/apps-up4/ninferno-serve /root/gbench/srv_any.sh start needle /opt/models/artemis/artemis-v1.2-imw-qkv-vision-nvfp4.r31.ninferno --vision --max-context 203136 --prefill-chunk 1280 --kv-dtype fp8 --sliding-kv-dtype fp8 --spec mtp --drafter /opt/models/dl2/assistant.nvfp4.ninferno --draft-tokens 7 --chat-template /opt/models/templates/gemma4-agent.jinja > /dev/null 2>&1
(cd /root/gbench && python3 needle.py --label "N9 rested needle" --lengths 140000 --depths 0.5 2>&1 | grep '"L"' | cut -c1-160)
step "resting the GPU 5 min"; /root/gbench/srv.sh stop > /dev/null 2>&1; sleep 300
./measure9.sh new /opt/models/n9/assistant-artemis-ep$BEST.nvfp4.ninferno; grep -E "^(TG|CHAT|HARD50|HELDOUT .* (all|greedy|sampled):)|DIED" m-new.txt
back "done"
