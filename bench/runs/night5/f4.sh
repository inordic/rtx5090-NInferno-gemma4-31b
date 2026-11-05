#!/bin/bash
# f4: does chunk 1280 cost cache hits when the cache is full? Same setup but for the chunk, two seeds.
source <(sed -n "/^cd \/root\/gbench/,/^counts() {/p" n5/night5.sh | sed "\$d")
LAST=$(date -u -d "2026-11-05 04:30" +%s)
take_gpu F4 || exit 1
{ for seed in 5 6; do for c in 1280 1024; do
      ./srv.sh stop
      L=F4_c${c}_s$seed
      start $L $UP4 1 $(flags 203136 $c $DR4) && {
          grep -hE "pinning host state" logs/$L.log | cut -c26-80
          run stress python3 stress_entitlement.py --mode agent --rounds 300 --seed $seed
          python3 - "$L" <<"PY"
import re,sys
R=[]
for ln in open("logs/%s.log"%sys.argv[1],errors="ignore"):
    if not re.search(r"req#\d+ done",ln): continue
    m=re.search(r"prompt ([\d,]+) \| output ([\d,]+) \| cache ([\d,]+)",ln)
    if m: R.append(tuple(int(x.replace(",","")) for x in m.groups()))
p=sum(r[0] for r in R); k=sum(r[2] for r in R)
print("STRESS",sys.argv[1],"requests",len(R),"prompt",p,"cached",k,"new",p-k,"cached %.1f%%"%(100*k/p))
PY
      }
  done; done
} > $O/F4.txt 2>&1
give_back >> $O/F4.txt 2>&1
step F4_DONE
