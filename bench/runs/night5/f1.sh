#!/bin/bash
# f1: after night 5. Patch 19 and the NVFP4 assistant fixed; how should the freed memory be spent?
# Chunk sweep at the largest context that fits, the top context at chunk 1024, and the CUDA 13.4 build.
source <(sed -n "/^cd \/root\/gbench/,/^counts() {/p" n5/night5.sh | sed "\$d")
counts() {
    echo "COUNTS $1: privately $(grep -c "publishes privately" logs/$1.log) | request failure $(grep -c "request failure" logs/$1.log) | exits $(grep -cE "Engine failed|FATAL|underflow" logs/$1.log)"
}
LAST=$(date -u -d "2026-11-05 04:30" +%s)
UP134=$B/apps-134/ninferno-serve
fs() { TAIL=9 run speed python3 gbench.py --label "F1 $1" --pp 2048,8192,32768,65536 --tg 8192 --reps 3 --skip-sampled; }
cap() { grep -hE "^capacity|plan \| rung" logs/$1.log | tail -2 | cut -c1-170; }
take_gpu F1 || exit 1
{
  echo "--- top context at chunk 1024"
  start_fit F1_top $UP4 1 1024 $DR4 217088 215040 213312 211968 210944 209920 208896 207872 && { cap F1_top; }
  for c in 1280 1152 1408 1536; do
      echo "--- chunk $c"
      if start_fit F1_c$c $UP4 1 $c $DR4 206848 204800 202112 200704 198656 196608; then cap F1_c$c; fs c$c; counts F1_c$c; fi
  done
  echo "--- chunk 1024"
  start_fit F1_c1024 $UP4 1 1024 $DR4 206848 && { cap F1_c1024; fs c1024; counts F1_c1024; }
  if [ -x $UP134 ]; then
      echo "--- cuda 13.4"
      ./srv.sh stop
      start F1_134 $UP134 1 $(flags 206848 1024 $DR4) && {
          fs cuda134; run ab python3 $O/n5ab.py run $O/ab_134.json; counts F1_134; }
      ./srv.sh stop
      start F1_132 $UP4 1 $(flags 206848 1024 $DR4) && { run ab python3 $O/n5ab.py run $O/ab_132.json; }
      python3 $O/n5ab.py cmp $O/ab_132.json $O/ab_134.json 2>&1 | tail -3 | cut -c1-200
  fi
} > $O/F1.txt 2>&1
give_back >> $O/F1.txt 2>&1
step F1_DONE
