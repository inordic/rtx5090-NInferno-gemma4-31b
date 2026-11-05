#!/bin/bash
# f2: the exact setup to deploy, through every gate. f3: production against it, turn about.
source <(sed -n "/^cd \/root\/gbench/,/^counts() {/p" n5/night5.sh | sed "\$d")
counts() {
    echo "COUNTS $1: privately $(grep -c "publishes privately" logs/$1.log) | request failure $(grep -c "request failure" logs/$1.log) | exits $(grep -cE "Engine failed|FATAL|underflow" logs/$1.log)"
    grep -hE "Engine failed|FATAL|underflow" logs/$1.log | head -2 | cut -c1-300
    grep -oE "mtp accepted [0-9]+/[0-9]+" logs/$1.log | tr "/" " " | awk -v L=$1 "{x += \$3; y += \$4} END {if (y) printf \"MTP %s accepted %d/%d (%.1f%%)\n\", L, x, y, 100 * x / y}"
}
LAST=$(date -u -d "2026-11-05 04:30" +%s)
CTX=203136; CH=1280
gpu() { nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tr -d "\n"; echo; }
if take_gpu F2; then
    { start F2_final $UP4 1 $(flags $CTX $CH $DR4) && {
          grep -hE "^capacity|capacity \|" logs/F2_final.log | tail -1 | cut -c1-170
          echo "GPU at start: $(gpu)"
          run verify python3 verify.py && run textcall python3 textcalltest.py \
              && run vision python3 visiontest.py --label "F2 final" && run image python3 $O/n5img.py \
              && run turns python3 turnprobe.py http://127.0.0.1:8080 \
              && run head python3 headprobe.py && run agent python3 agenttest.py --label "F2 final" --runs 1 \
              && run replay1 python3 cachereplay.py --label "F2 final r1" --slots 12 \
              && run replay2 python3 cachereplay.py --label "F2 final r2" --slots 12 \
              && run replay3 python3 cachereplay.py --label "F2 final r3" --slots 12 \
              && run stress python3 stress_entitlement.py --mode agent --rounds 300 --seed 5 \
              && TAIL=4 run needle python3 needle.py --label "F2 final needle" --lengths 140000,170000,205000 --depths 0.5 \
              && TAIL=6 run quality python3 geval.py --label "F2-final" --suites mmlupro,mmlu,gsm8k,humaneval --conc 1 \
              && run hard50 python3 $HS --url http://127.0.0.1:8080/v1 --set $SET --tag f2_final --out hard50/f2_final.json
          if alive; then echo "SURVIVED F2_final"; else echo "EXITED F2_final"; fi
          echo "GPU at end: $(gpu)"
          counts F2_final
      }
    } > $O/F2.txt 2>&1
    give_back >> $O/F2.txt 2>&1
    step F2_DONE
fi
fsp() { TAIL=9 run speed python3 gbench.py --label "F3 $1" --pp $PP --tg 8192 --reps 2 --skip-sampled; }
if grep -q "SURVIVED F2_final" $O/F2.txt && take_gpu F3; then
    { for r in 1 2; do
          ./srv.sh stop
          start F3_before$r $UP3 0 $(flags 202112 1024 $DR) && { fsp before$r; counts F3_before$r; }
          ./srv.sh stop
          start F3_after$r $UP4 1 $(flags $CTX $CH $DR4) && { fsp after$r; counts F3_after$r; }
      done
    } > $O/F3.txt 2>&1
    give_back >> $O/F3.txt 2>&1
    step F3_DONE
fi
step F_ALL_DONE
