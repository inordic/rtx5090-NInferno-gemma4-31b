#!/bin/bash
# late10.sh: night 10's activation-scale KL, after gpu10.sh has finished and restored production.
# gpu10.sh's Q5_K_M reference could not start: the Q5_K_M file stores some layers' gate and up matrices in different
# formats (Q8_0 and Q5_K), and the engine needs them as one packed plane. So the reference here is production's own
# artifact with exact (BF16) activations and BF16 KV. V0, V1 and V2 have bit-identical weights, so KL against it
# measures only the 4-bit activations, which is all the calibration changes. A sanity run checks that exact
# activations ignore the scales (V1 exact against the V0 reference should be ~0).
# Then, if a calibrated version wins and time allows: hard50, verify and the text tool-call test on it.
# Production is stopped only when idle and restarted unchanged by END; watchdog10.sh still covers it at 05:32.
exec 9> /root/gbench/n10/late10.lock; flock -n 9 || { echo "late10 already running"; exit 1; }
cd /root/gbench/n10
echo $$ > late10.pid
END_TS=$(date -u -d "${END:-2026-11-14 05:25}" +%s)
A0=/opt/models/artemis/artemis-v1.2-imw-qkv-vision-nvfp4.r31.ninferno
D9=/opt/models/artemis/assistant-artemis-n9.nvfp4.ninferno
export NINFERNO_GEMMA4_SHARED_VALUES=2 NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_EMBED_HOST=1 NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64 NINFERNO_GEMMA4_SKIP_LAST_LAYER=1
PROD="--vision --max-context 203136 --prefill-chunk 1280 --kv-dtype fp8 --sliding-kv-dtype fp8 --spec mtp --chat-template /opt/models/templates/gemma4-agent.jinja --auto-tools-shared-prefix --auto-shared-prefix-requested"
step() { echo "=== $(date -u +%T) $*"; }
left() { echo $(( END_TS - $(date -u +%s) )); }
need() { [ "$(left)" -gt "$1" ] || { step "SKIPPED (time): $2"; return 1; }; }
back() {
    step "production back ($1)"; ./srv10.sh stop > /dev/null 2>&1; systemctl start gemma4-31b-fast
    for i in $(seq 1 90); do curl -sf -m 5 http://127.0.0.1:8080/v1/models > /dev/null && break; sleep 5; done
    step "production $(systemctl is-active gemma4-31b-fast)"
    (cd /root/gbench && timeout 300 python3 verify.py --model gemma4-31b-fast > n10/late_verify.out 2>&1); tail -1 late_verify.out
    (cd /root/gbench && timeout 120 python3 textcalltest.py > n10/late_textcall.out 2>&1); tail -1 late_textcall.out
    step "LATE10_END: $1"; exit 0
}

until grep -q GPU10_END gpu10.out; do sleep 20; done
step "gpu10 finished: $(grep GPU10_END gpu10.out)"
need 2400 "late KL (production stays up)" || exit 0
sleep 30  # let gpu10's own checks settle
for i in $(seq 1 120); do
    a=$(curl -s -m 5 http://127.0.0.1:8080/health | python3 -c "import json,sys; print(json.load(sys.stdin)['active_requests'])" 2>/dev/null)
    [ "$a" = "0" ] && break; sleep 5
done
step "stopping production"; systemctl stop gemma4-31b-fast; ./srv10.sh stop > /dev/null 2>&1; sleep 3

SANITY=0
for t in code agent chat; do
    need 1500 "KL reference $t" || break
    ./kl10b.sh ref $A0 kl_$t.txt /opt/models/n10/refp_$t.kld pref_$t
    [ -s /opt/models/n10/refp_$t.kld ] || continue
    if [ $SANITY = 0 ]; then SANITY=1; need 1200 "sanity" && ./kl10b.sh exact /opt/models/n10/V1.ninferno kl_$t.txt /opt/models/n10/refp_$t.kld pV1exact_$t; fi
    for v in V0 V1 V2; do
        art=$A0; [ $v != V0 ] && art=/opt/models/n10/$v.ninferno
        need 600 "KL $v $t" && ./kl10b.sh run $art kl_$t.txt /opt/models/n10/refp_$t.kld p${v}_$t
    done
done

WIN=$(python3 - <<'EOF'
import glob, re
res = {}
for f in glob.glob("/root/gbench/n10/kld/pV[012]_*.log"):
    m = re.search(r"mean KLD: ([\d.]+)", open(f).read())
    if m:
        res.setdefault(f.split("/")[-1].split("_")[0][1:], []).append(float(m.group(1)))
full = {v: sum(x) / len(x) for v, x in res.items() if len(x) == len(res.get("V0", []))}
print(min(full, key=full.get) if full else "V0")
EOF
)
step "KL winner (production's own exact activations as reference): $WIN"
if [ "$WIN" != V0 ] && need 1700 "quality checks for $WIN"; then
    ./srv10.sh stop > /dev/null 2>&1; sleep 30
    SRV_BIN=/opt/NInferno-up/build/apps-up4/ninferno-serve ./srv10.sh start q-$WIN /opt/models/n10/$WIN.ninferno $PROD --drafter $D9 --draft-tokens 7 > srv-q.out 2>&1 && {
        step "quality checks $WIN"
        (cd /root/gbench && timeout 900 python3 /opt/NInferno-up/tools/bench/hard_set.py --url http://127.0.0.1:8080/v1 --set /opt/NInferno-up/eval/corpora/hard50/hard50.json --tag n10_$WIN --out hard50/n10_$WIN.json > n10/q-$WIN.hard50.txt 2>&1)
        python3 -c "
import json; r=json.load(open('/root/gbench/hard50/n10_$WIN.json'))['rows']; print('HARD50 $WIN', sum(1 for x in r if x['correct']), '/', len(r))"
        need 420 "verify $WIN" && { (cd /root/gbench && timeout 300 python3 verify.py > n10/q-$WIN.verify.out 2>&1); tail -1 q-$WIN.verify.out; }
        need 240 "textcall $WIN" && { (cd /root/gbench && timeout 120 python3 textcalltest.py > n10/q-$WIN.textcall.out 2>&1); tail -1 q-$WIN.textcall.out; }
    }
fi
back "done"
