#!/bin/bash
# fix10.sh: night 10's corrected finish, started 04:07 UTC in place of the rest of gpu10.sh and late10.sh.
# srv10.sh stop required SRV_BIN, which gpu10.sh set only for start, so every stop silently did nothing: the A/B
# server ab-n9a kept the GPU, the servers for n10a, n9b, n10b and n10k5 failed to start (out of memory), and their
# requests were answered by ab-n9a's server. That gives four runs of the n9 drafter, all in logs/ab-n9a.log, and none
# of n10. Fixed (stop no longer needs SRV_BIN). Now: two runs of the n10 drafter, then the KL of late10.sh
# (production's own exact activations as the reference) as far as time allows, then production back by END.
exec 9> /root/gbench/n10/fix10.lock; flock -n 9 || { echo "fix10 already running"; exit 1; }
cd /root/gbench/n10
echo $$ > fix10.pid
END_TS=$(date -u -d "${END:-2026-11-14 05:25}" +%s)
A0=/opt/models/artemis/artemis-v1.2-imw-qkv-vision-nvfp4.r31.ninferno
D10=/opt/models/n10/assistant-artemis-n10.nvfp4.ninferno
export NINFERNO_GEMMA4_SHARED_VALUES=2 NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_EMBED_HOST=1 NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64 NINFERNO_GEMMA4_SKIP_LAST_LAYER=1
PROD="--vision --max-context 203136 --prefill-chunk 1280 --kv-dtype fp8 --sliding-kv-dtype fp8 --spec mtp --chat-template /opt/models/templates/gemma4-agent.jinja --auto-tools-shared-prefix --auto-shared-prefix-requested"
step() { echo "=== $(date -u +%T) $*"; }
left() { echo $(( END_TS - $(date -u +%s) )); }
need() { [ "$(left)" -gt "$1" ] || { step "SKIPPED (time): $2"; return 1; }; }
back() {
    step "production back ($1)"; ./srv10.sh stop > /dev/null 2>&1
    pgrep -f "apps[^ ]*/ninferno-serve" > /dev/null && { step "a test server is still running; killing it"; pkill -9 -f "apps[^ ]*/ninferno-serve"; sleep 3; }
    systemctl start gemma4-31b-fast
    for i in $(seq 1 90); do curl -sf -m 5 http://127.0.0.1:8080/v1/models > /dev/null && break; sleep 5; done
    step "production $(systemctl is-active gemma4-31b-fast)"
    (cd /root/gbench && timeout 300 python3 verify.py --model gemma4-31b-fast > n10/fix_verify.out 2>&1); tail -1 fix_verify.out
    (cd /root/gbench && timeout 120 python3 textcalltest.py > n10/fix_textcall.out 2>&1); tail -1 fix_textcall.out
    step "FIX10_END: $1"; exit 0
}
ab() {  # ab LABEL DRAFTER K
    ./srv10.sh stop > /dev/null 2>&1; sleep 60
    pgrep -f "apps[^ ]*/ninferno-serve" > /dev/null && { step "$1: a server is still running, skipped"; return 1; }
    SRV_BIN=/opt/NInferno-up/build/apps-up4/ninferno-serve ./srv10.sh start $1 $A0 $PROD --drafter $2 --draft-tokens $3 > srv-$1.out 2>&1 || { step "$1 DIED"; tail -3 srv-$1.out; return 1; }
    grep -q "listening on" /root/gbench/logs/$1.log || { step "$1 never listened"; return 1; }
    grep -E "drafter|speculation" /root/gbench/logs/$1.log | head -2 | cut -c1-200
    timeout 1500 python3 heldout10.py $1 /root/gbench/logs/$1.log > ab-$1.txt 2>&1
    grep -E "^HELDOUT .* (all|greedy|sampled):|WARNING" ab-$1.txt
}

mkdir -p broken; for l in ab-n10a ab-n9b ab-n10b; do mv -f heldout-$l.json ab-$l.txt srv-$l.out broken/ 2>/dev/null; mv -f /root/gbench/logs/$l.log broken/ 2>/dev/null; done
for L in n10a n10b; do
    need 2100 "A/B $L" || break
    step "A/B $L"; ab ab-$L $D10 7
done

./srv10.sh stop > /dev/null 2>&1; sleep 3
SANITY=0
for t in code agent chat; do
    need 1200 "KL reference $t" || break
    ./kl10b.sh ref $A0 kl_$t.txt /opt/models/n10/refp_$t.kld pref_$t
    [ -s /opt/models/n10/refp_$t.kld ] || continue
    if [ $SANITY = 0 ]; then SANITY=1; need 900 "sanity" && ./kl10b.sh exact /opt/models/n10/V1.ninferno kl_$t.txt /opt/models/n10/refp_$t.kld pV1exact_$t; fi
    for v in V0 V1 V2; do
        art=$A0; [ $v != V0 ] && art=/opt/models/n10/$v.ninferno
        need 420 "KL $v $t" && ./kl10b.sh run $art kl_$t.txt /opt/models/n10/refp_$t.kld p${v}_$t
    done
done
back "done"
