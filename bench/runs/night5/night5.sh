#!/bin/bash
# night5: windows behind the idle gate, the fast unit back after each; no window starts after 05:30
# CET and a window stops when its server is gone. Production stays as it is.
#  W1 base    apps-up3 as deployed: speed and a power trace; prefill chunk 1280 and 1792
#  W2 ab      apps-up4 (six upstream fixes, patch 19): greedy answers off, off, on; speed off and on
#  W3 gates   apps-up4 with patch 19 on: replays, stress, tools, vision, needles, quality
#  W4 skip    the counting build: how many tiles skip-softmax could drop
#  W5 p0      the CudaNoStablePerfLimit profile, removed after the window
#  W6 kld     ninferno-perplexity: where the 4-bit error sits
#  W7 drafter the assistant in NVFP4
#  W8 cuda    the same tree built with CUDA 13.4, only with disk to spare
cd /root/gbench
O=n5; mkdir -p $O logs hard50
step() { echo "=== $(date +%T) $*"; }
LAST=$(date -u -d '2026-11-05 04:30' +%s)
export NINFERNO_GEMMA4_SHARED_VALUES=2 NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1
B=/opt/NInferno-up/build
UP3=$B/apps-up3/ninferno-serve; UP4=$B/apps-up4/ninferno-serve; SK=$B/apps-sk/ninferno-serve
LILA=/opt/models/dl2/turbo-qkv-vision-nvfp4.r31.ninferno
DR=/opt/models/dl2/assistant.q8full.ninferno; DR4=/opt/models/dl2/assistant.nvfp4.ninferno
TPL=/opt/models/templates/gemma4-agent.jinja
HS=/opt/NInferno-up/tools/bench/hard_set.py; SET=/opt/NInferno-up/eval/corpora/hard50/hard50.json
PROFILE=/etc/nvidia/nvidia-application-profiles-rc; PROFILE2=/root/.nv/nvidia-application-profiles-rc
PP=2048,8192,32768,65536,131072
flags() {  # context chunk drafter
    echo "--vision --max-context $1 --prefill-chunk $2 --kv-dtype fp8 --sliding-kv-dtype fp8 --spec mtp --drafter $3 --draft-tokens 7 --chat-template $TPL --auto-tools-shared-prefix --auto-shared-prefix-requested --host-state-slots 18 --host-kv-mib 8192"
}
take_gpu() {
    [ "$(date -u +%s)" -ge $LAST ] && { step "$1: too late to start, skipped"; return 1; }
    ./idle_gate_any.sh 5
    [ "$(date -u +%s)" -ge $LAST ] && { step "$1: too late after the gate, skipped"; return 1; }
    step "$1: idle, taking the GPU"
    systemctl stop gemma4-31b-fast
    for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
}
give_back() {
    trace_off; ./srv.sh stop; rm -f $PROFILE $PROFILE2
    step "restoring the fast unit"; ./deploy_fast.sh
}
alive() { pgrep -f "apps[^ ]*/ninferno-serve" > /dev/null && curl -sf -m 10 http://127.0.0.1:8080/v1/models > /dev/null; }
run() {
    alive || { echo "--- $1: server gone, window stopped"; return 1; }
    echo "--- $1"; shift; "$@" 2>&1 | tail -${TAIL:-3} | cut -c1-260
}
start() {  # label binary skip-last-layer flags...
    local L=$1 BIN=$2 SKIP=$3; shift 3
    NINFERNO_GEMMA4_EMBED_HOST=1 NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64 NINFERNO_GEMMA4_SKIP_LAST_LAYER=$SKIP \
        SRV_BIN=$BIN ./srv_any.sh start $L $LILA "$@" 2>&1 | tail -4 | cut -c1-220
    alive
}
start_fit() {  # label binary skip chunk drafter contexts...: the first context the plan accepts
    local L=$1 BIN=$2 SKIP=$3 CHUNK=$4 DRF=$5; shift 5
    for ctx in "$@"; do
        ./srv.sh stop
        if start $L $BIN $SKIP $(flags $ctx $CHUNK $DRF); then echo "FIT $L chunk $CHUNK context $ctx"; return 0; fi
        echo "no fit: $L chunk $CHUNK context $ctx"
    done
    return 1
}
trace_on() {
    TRACE=$O/trace_$1.csv
    nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,enforced.power.limit,temperature.gpu,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.hw_power_brake_slowdown --format=csv,noheader -lms 500 > $TRACE &
    TPID=$!
}
trace_off() { [ -n "$TPID" ] && { kill $TPID 2>/dev/null; wait $TPID 2>/dev/null; TPID=; python3 $O/n5trace.py $TRACE; }; }
speed() { TAIL=${2:-9} run speed python3 gbench.py --label "N5 $1" --pp $PP --tg 8192 --reps 2 --skip-sampled; }
counts() {
    echo "COUNTS $1: privately $(grep -c 'publishes privately' logs/$1.log) | request failure $(grep -c 'request failure' logs/$1.log) | exits $(grep -cE 'Engine failed|FATAL|underflow' logs/$1.log)"
    grep -hE "Engine failed|FATAL|underflow" logs/$1.log | head -2 | cut -c1-300
    grep -oE "mtp accepted [0-9]+/[0-9]+" logs/$1.log | tr '/' ' ' | awk -v L=$1 '{x += $3; y += $4} END {if (y) printf "MTP %s accepted %d/%d (%.1f%%)\n", L, x, y, 100 * x / y}'
}

step "assistant in NVFP4, on the CPU"
[ -e $DR4 ] || nice -n 15 $B/apps/ninferno-convert --model /opt/models/dl2/assistant --recipe gemma4_assistant_published \
    --drafter-quant nvfp4 --out $DR4 2>&1 | tail -2 | cut -c1-200
ls -la $DR4 2>&1 | cut -c1-120

if take_gpu W1; then
    { trace_on base
      start N5_base $UP3 0 $(flags 202112 1024 $DR) && speed base 13
      trace_off
      counts N5_base
      for c in 1280 1792; do
          if start_fit N5_c$c $UP3 0 $c $DR 202112 198656 196608 194560 192512 190464 188416 184320 180224; then
              trace_on c$c; speed c$c 8; trace_off; counts N5_c$c
          fi
      done
    } > $O/W1.txt 2>&1
    give_back >> $O/W1.txt 2>&1
    grep -hE "^(FIT|TRACE|COUNTS|PP )" $O/W1.txt | cut -c1-240
fi

if take_gpu W2; then
    { start N5_ab0 $UP4 0 $(flags 202112 1024 $DR) && { run ab-off python3 $O/n5ab.py run $O/ab_off.json; speed up4off 8; counts N5_ab0; }
      ./srv.sh stop
      start N5_ab1 $UP4 0 $(flags 202112 1024 $DR) && { run ab-off2 python3 $O/n5ab.py run $O/ab_off2.json; counts N5_ab1; }
      ./srv.sh stop
      start N5_ab2 $UP4 1 $(flags 202112 1024 $DR) && { run ab-on python3 $O/n5ab.py run $O/ab_on.json; speed up4on 8; counts N5_ab2; }
      python3 $O/n5ab.py cmp $O/ab_off.json $O/ab_off2.json 2>&1 | tail -4 | cut -c1-220
      python3 $O/n5ab.py cmp $O/ab_off.json $O/ab_on.json 2>&1 | tail -4 | cut -c1-220
    } > $O/W2.txt 2>&1
    give_back >> $O/W2.txt 2>&1
    grep -hE "^(same|differs|COUNTS|PP )" $O/W2.txt | cut -c1-220
fi

if take_gpu W3; then
    { (cd $B && for t in ninferno_qwen3_5_state_image_test ninferno_qwen3_5_context_store_test; do
          ./tests/$t > /root/gbench/$O/unit_$t.log 2>&1 && echo "PASS $t" || { echo "FAIL $t"; tail -3 /root/gbench/$O/unit_$t.log | cut -c1-200; }
      done)
      start N5_gates $UP4 1 $(flags 202112 1024 $DR) && {
          run verify python3 verify.py && run textcall python3 textcalltest.py \
              && run vision python3 visiontest.py --label "N5 up4" && run image python3 $O/n5img.py \
              && run turns python3 turnprobe.py http://127.0.0.1:8080 \
              && run head python3 headprobe.py && run agent python3 agenttest.py --label "N5 up4" --runs 1 \
              && run replay1 python3 cachereplay.py --label "N5 up4 r1" --slots 12 \
              && run replay2 python3 cachereplay.py --label "N5 up4 r2" --slots 12 \
              && run replay3 python3 cachereplay.py --label "N5 up4 r3" --slots 12 \
              && run stress python3 stress_entitlement.py --mode agent --rounds 300 --seed 5 \
              && run needle python3 needle.py --label "N5 up4 needle" --lengths 140000,170000 --depths 0.5 \
              && run quality python3 geval.py --label "N5-up4" --suites mmlupro,gsm8k --conc 1 \
              && run hard50 python3 $HS --url http://127.0.0.1:8080/v1 --set $SET --tag n5_up4 --out hard50/n5_up4.json
          if alive; then echo "SURVIVED N5_gates"; else echo "EXITED N5_gates"; fi
          counts N5_gates
      }
    } > $O/W3.txt 2>&1
    give_back >> $O/W3.txt 2>&1
    grep -hE "^(PASS|FAIL|SURVIVED|EXITED|COUNTS|MTP)|status " $O/W3.txt | cut -c1-220
fi

if take_gpu W4; then
    { start N5_skip $SK 0 $(flags 202112 1024 $DR) && TAIL=6 run skip python3 $O/n5skip.py logs/N5_skip.log 8192 32768 65536 131072
      TAIL=4 run skip2 python3 $O/n5skip.py logs/N5_skip.log 32768 131072
    } > $O/W4.txt 2>&1
    give_back >> $O/W4.txt 2>&1
    grep -h "SKIP" $O/W4.txt | cut -c1-200
fi

if [ ! -e $PROFILE ] && [ ! -e $PROFILE2 ] && take_gpu W5; then
    { mkdir -p /etc/nvidia /root/.nv
      cat > $PROFILE <<'JSON'
{
  "profiles": [ { "name": "CudaNoStablePerfLimit", "settings": ["0x166c5e", 0] } ],
  "rules":    [ { "pattern": { "feature": "procname", "matches": "ninferno-serve" },
                  "profile": "CudaNoStablePerfLimit" } ]
}
JSON
      cp $PROFILE $PROFILE2
      trace_on p0
      start N5_p0 $UP4 1 $(flags 202112 1024 $DR) && speed p0 13
      trace_off
      counts N5_p0
    } > $O/W5.txt 2>&1
    give_back >> $O/W5.txt 2>&1
    ls $PROFILE $PROFILE2 2>&1 | cut -c1-100
    grep -hE "^(TRACE|PP |TG|CHAT)" $O/W5.txt | cut -c1-240
fi

if take_gpu W6; then
    { cd /opt/NInferno-up
      P="timeout 1200 $B/apps/ninferno-perplexity $LILA --corpus eval/corpora/perplexity-1m/manifest.json --quick --context 4096 --stride 2048"
      export NINFERNO_GEMMA4_EMBED_HOST=1 NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64
      K=/root/gbench/$O/a16.kld
      echo "--- reference: weights only"; $P --activations exact --kv-dtype bf16 --kld-out $K 2>&1 | tail -4 | cut -c1-220
      if [ -s $K ]; then
          echo "--- production: A4 + FP8 KV"; $P --kv-dtype fp8 --kld $K 2>&1 | tail -8 | cut -c1-220
          echo "--- KV alone"; $P --activations exact --kv-dtype fp8 --kld $K 2>&1 | tail -8 | cut -c1-220
          echo "--- A4 alone"; $P --kv-dtype bf16 --kld $K 2>&1 | tail -8 | cut -c1-220
      fi
      cd /root/gbench
    } > $O/W6.txt 2>&1
    give_back >> $O/W6.txt 2>&1
    grep -hiE "^---|kld|top-1|ppl" $O/W6.txt | cut -c1-200 | tail -24
fi

if [ -s $DR4 ] && take_gpu W7; then
    { start_fit N5_dr4 $UP4 1 1024 $DR4 206848 204800 202112 && {
          speed dr4 9
          run hard50 python3 $HS --url http://127.0.0.1:8080/v1 --set $SET --tag n5_dr4 --out hard50/n5_dr4.json
          counts N5_dr4
      }
    } > $O/W7.txt 2>&1
    give_back >> $O/W7.txt 2>&1
    grep -hE "^(FIT|COUNTS|MTP|PP |TG|CHAT)" $O/W7.txt | cut -c1-220
fi

free_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
if [ "$free_gb" -ge 8 ] && [ "$(date -u +%s)" -lt $LAST ]; then
    step "CUDA 13.4 aside ($free_gb GB free)"
    { apt-get install -y -q cuda-minimal-build-13-4 2>&1 | tail -1 | cut -c1-160
      cd /opt/NInferno-up
      PATH=/usr/local/cuda-13.4/bin:$PATH cmake -S . -B build-134 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a \
          -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.4/bin/nvcc -DBUILD_TESTING=OFF -DNINFERNO_BUILD_BENCHMARKS=OFF 2>&1 | tail -1 | cut -c1-160
      PATH=/usr/local/cuda-13.4/bin:$PATH nice -n 15 ninja -C build-134 -j6 ninferno-serve > /root/gbench/$O/build_134.log 2>&1 \
          && mkdir -p build/apps-134 && cp build-134/apps/ninferno-serve build/apps-134/ && echo BUILD_134_OK \
          || { echo BUILD_134_FAILED; grep -m4 -E "error|FAILED" /root/gbench/$O/build_134.log | cut -c1-200; }
      rm -rf build-134
      cd /root/gbench
    } > $O/W8.txt 2>&1
    grep -hE "BUILD_134" $O/W8.txt
    if [ -x $B/apps-134/ninferno-serve ] && take_gpu W8; then
        { start N5_134 $B/apps-134/ninferno-serve 1 $(flags 202112 1024 $DR) && {
              speed cuda134 13
              run ab python3 $O/n5ab.py run $O/ab_134.json
              python3 $O/n5ab.py cmp $O/ab_on.json $O/ab_134.json 2>&1 | tail -3 | cut -c1-200
              counts N5_134
          }
        } >> $O/W8.txt 2>&1
        give_back >> $O/W8.txt 2>&1
        grep -hE "^(same|COUNTS|PP |TG|CHAT)" $O/W8.txt | cut -c1-220
    fi
fi
curl -sf http://127.0.0.1:8080/v1/models > /dev/null && step "fast unit answering: $(systemctl is-active gemma4-31b-fast)"
df -h / | tail -1
step NIGHT5_DONE
