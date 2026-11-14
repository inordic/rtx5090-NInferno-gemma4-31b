#!/bin/bash
# gpu10.sh: night 10's GPU phase. After collection (all jobs done, but not before START), with production stopped:
#   1. capture Artemis's states for the new replies, with activation peaks over all of them
#   2. train the drafter n10 from night 9's (new data plus half of night 9's as replay), one pass
#      meanwhile on the CPU: calibrated activation scales V1/V2 and their artifacts, the KL texts
#   3. KL against Artemis's Q5_K_M (exact activations) for production's scales (V0), V1 and V2, on three texts
#   4. drafter A/B on production's binary and flags, alternating n9 / n10 / n9 / n10
#   5. if time: the KL winner's quality checks (only if it is not V0), then draft depth 5 for both drafters
# Every step checks the clock; production is restarted unchanged by END at the latest, and a separate watchdog
# (watchdog10.sh) restarts it if this script does not. Test servers listen on 127.0.0.1 only.
exec 9> /root/gbench/n10/gpu10.lock; flock -n 9 || { echo "gpu10 already running"; exit 1; }
cd /root/gbench/n10
echo $$ > gpu10.pid
START_TS=$(date -u -d "${START:-2026-11-14 01:30}" +%s)
END_TS=$(date -u -d "${END:-2026-11-14 05:25}" +%s)
PY=/opt/venv-cu/bin/python
A0=/opt/models/artemis/artemis-v1.2-imw-qkv-vision-nvfp4.r31.ninferno
D9=/opt/models/artemis/assistant-artemis-n9.nvfp4.ninferno
D10=/opt/models/n10/assistant-artemis-n10.nvfp4.ninferno
REF=/opt/models/n10/ref/a2q5.ninferno
export NINFERNO_GEMMA4_SHARED_VALUES=2 NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_EMBED_HOST=1 NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64 NINFERNO_GEMMA4_SKIP_LAST_LAYER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PROD="--vision --max-context 203136 --prefill-chunk 1280 --kv-dtype fp8 --sliding-kv-dtype fp8 --spec mtp --chat-template /opt/models/templates/gemma4-agent.jinja --auto-tools-shared-prefix --auto-shared-prefix-requested"
step() { echo "=== $(date -u +%T) $*"; }
left() { echo $(( END_TS - $(date -u +%s) )); }
need() { [ "$(left)" -gt "$1" ] || { step "SKIPPED (time): $2"; return 1; }; }
back() {
    step "production back ($1)"; ./srv10.sh stop > /dev/null 2>&1; systemctl start gemma4-31b-fast
    for i in $(seq 1 90); do curl -sf -m 5 http://127.0.0.1:8080/v1/models > /dev/null && break; sleep 5; done
    step "production $(systemctl is-active gemma4-31b-fast)"
    (cd /root/gbench && timeout 300 python3 verify.py --model gemma4-31b-fast > n10/back_verify.out 2>&1); tail -1 back_verify.out
    (cd /root/gbench && timeout 120 python3 textcalltest.py > n10/back_textcall.out 2>&1); tail -1 back_textcall.out
    step "GPU10_END: $1"; exit 0
}

# ---- wait for the collection
until grep -q GEN_DONE gen.out 2>/dev/null || [ "$(date -u +%s)" -ge "$START_TS" ]; do sleep 30; done
until [ "$(date -u +%s)" -ge "$START_TS" ]; do sleep 30; done
pkill -INT -f "[g]en10.py run"; sleep 10
step "collected $(wc -l < /opt/models/n10/gen.jsonl) replies ($(grep -c GEN_DONE gen.out) GEN_DONE)"
$PY -I fwd10.py tokenize 2>&1 | grep TOKENIZED
for i in $(seq 1 240); do
    a=$(curl -s -m 5 http://127.0.0.1:8080/health | python3 -c "import json,sys; print(json.load(sys.stdin)['active_requests'])" 2>/dev/null)
    [ "$a" = "0" ] && break; sleep 5
done
step "stopping production"; systemctl stop gemma4-31b-fast; ./srv10.sh stop > /dev/null 2>&1; sleep 3

# ---- 1. capture
step "capture"
timeout 4800 $PY -I fwd10.py run imw cuda > fwd.log 2>&1 && CAP=1 || { CAP=0; tail -5 fwd.log; }
grep -E "pass .* written|FWD_DONE" fwd.log | tail -2

# ---- CPU, in the background: calibrated scales, their artifacts, the KL texts
(
  AM="/opt/models/n9/amax-imw.json"; [ "$CAP" = 1 ] && AM="$AM /opt/models/n10/amax-imw.json"
  nice -n 10 $PY -I calib10.py /opt/models/n9/A2-q-imw /opt/models/n10/V1q /opt/models/n10/V2q $AM > calib.log 2>&1
  for v in V1 V2; do ./conv.sh /opt/models/n10/${v}q /opt/models/n8/A2/bf16 /opt/models/n10/$v.ninferno >> calib.log 2>&1; done
  nice -n 10 $PY -I kltext10.py >> calib.log 2>&1
  echo CPU_DONE >> calib.log
) &
CPUJOB=$!

# ---- 2. training
if [ "$CAP" = 1 ] && need 7200 "training"; then
    step "training n10"
    ACCUM=4096 TRAIN_CHUNK=512 timeout $(( $(left) - 6600 )) $PY -I train10.py train /opt/models/n10/states-imw:1.0,/opt/models/n9/states-imw:0.5 /opt/models/n10/asst 1 3e-5 > train.log 2>&1 \
        && TR=1 || { TR=0; tail -3 train.log; }
    grep -E "base drafter|train:|epoch [0-9]+ \[" train.log | cut -c1-330
    if [ "$TR" = 1 ]; then
        /opt/NInferno-up/build/apps/ninferno-convert --model /opt/models/n10/asst-ep1 --recipe gemma4_assistant_published --drafter-quant nvfp4 \
            --out $D10 > convert.log 2>&1 && ls -la $D10 || { TR=0; tail -2 convert.log; }
    fi
else TR=0; fi

# ---- 3. KL
step "waiting for the CPU jobs"; wait $CPUJOB; grep -E "clip under|wrote|CONV_DONE|CONV_FAILED|kl_.*sequences|CPU_DONE" calib.log
for t in chat code agent; do
    need 3600 "KL reference $t" || break
    ./kld10.sh ref $REF kl_$t.txt /opt/models/n10/ref_$t.kld
done
for v in V0 V1 V2; do
    art=$A0; [ $v != V0 ] && art=/opt/models/n10/$v.ninferno
    [ -f $art ] || { step "no artifact for $v"; continue; }
    for t in chat code agent; do
        [ -f /opt/models/n10/ref_$t.kld ] || continue
        need 3300 "KL $v $t" && ./kld10.sh run $art kl_$t.txt /opt/models/n10/ref_$t.kld ${v}_$t
    done
done

# ---- 4. drafter A/B
ab() {  # ab LABEL ARTIFACT DRAFTER K
    ./srv10.sh stop > /dev/null 2>&1; sleep 90
    SRV_BIN=/opt/NInferno-up/build/apps-up4/ninferno-serve ./srv10.sh start $1 $2 $PROD --drafter $3 --draft-tokens $4 > srv-$1.out 2>&1 || { step "$1 DIED"; cat srv-$1.out | tail -3; return 1; }
    timeout 1500 python3 heldout10.py $1 /root/gbench/logs/$1.log > ab-$1.txt 2>&1
    grep -E "^HELDOUT .* (all|english|other-languages|greedy|sampled|code|maths|agent|agent-later):|WARNING" ab-$1.txt
}
if [ "$TR" = 1 ]; then
    for L in n9a n10a n9b n10b; do
        need 2700 "A/B $L" || break
        case $L in n9*) d=$D9;; *) d=$D10;; esac
        step "A/B $L"; ab ab-$L $A0 $d 7
    done
fi

# ---- 5. extras, while there is time
WIN=$(python3 - <<'EOF'
import glob, re
best = {}
for f in glob.glob("/root/gbench/n10/kld/V*_*.log"):
    m = re.search(r"mean KLD: ([\d.]+)", open(f).read())
    if m:
        v = f.split("/")[-1].split("_")[0]
        best.setdefault(v, []).append(float(m.group(1)))
full = {v: sum(x) / len(x) for v, x in best.items() if len(x) == 3}
print(min(full, key=full.get) if full else "V0")
EOF
)
step "KL winner: $WIN"
if [ "$WIN" != V0 ] && need 2400 "quality checks for $WIN"; then
    ./srv10.sh stop > /dev/null 2>&1; sleep 60
    SRV_BIN=/opt/NInferno-up/build/apps-up4/ninferno-serve ./srv10.sh start q-$WIN /opt/models/n10/$WIN.ninferno $PROD --drafter $D9 --draft-tokens 7 > srv-q.out 2>&1 && {
        step "quality checks $WIN"
        (cd /root/gbench && timeout 900 python3 /opt/NInferno-up/tools/bench/hard_set.py --url http://127.0.0.1:8080/v1 --set /opt/NInferno-up/eval/corpora/hard50/hard50.json --tag n10_$WIN --out hard50/n10_$WIN.json > n10/q-$WIN.hard50.txt 2>&1)
        python3 -c "
import json; r=json.load(open('/root/gbench/hard50/n10_$WIN.json'))['rows']; print('HARD50 $WIN', sum(1 for x in r if x['correct']), '/', len(r))"
        (cd /root/gbench && timeout 300 python3 verify.py > n10/q-$WIN.verify.out 2>&1); tail -1 q-$WIN.verify.out
        (cd /root/gbench && timeout 120 python3 textcalltest.py > n10/q-$WIN.textcall.out 2>&1); tail -1 q-$WIN.textcall.out
        (cd /root/gbench && timeout 300 python3 visiontest.py --label "N10 $WIN" > n10/q-$WIN.vision.out 2>&1); tail -1 q-$WIN.vision.out
        (cd /root/gbench && timeout 360 python3 -I companionfit.py N10-$WIN > n10/q-$WIN.companion.out 2>&1); tail -1 q-$WIN.companion.out | cut -c1-160
        (cd /root/gbench && timeout 600 python3 agenttest.py --label "N10 $WIN" --runs 1 > n10/q-$WIN.agent.out 2>&1); grep "^AGENT" q-$WIN.agent.out | cut -c1-260
    }
fi
if [ "$TR" = 1 ]; then
    for L in n10k5 n9k5; do
        need 1800 "depth 5 $L" || break
        case $L in n9*) d=$D9;; *) d=$D10;; esac
        step "A/B $L (draft depth 5)"; ab ab-$L $A0 $d 5
    done
fi
back "done"
