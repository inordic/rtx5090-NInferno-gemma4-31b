#!/usr/bin/env python3
"""n5trace TRACE.csv [LABEL]: the GPU under load from an nvidia-smi trace: P-states, median and low
clocks, power, and how often each slowdown reason was active."""
import csv, statistics as st, sys

rows = [r for r in csv.reader(open(sys.argv[1])) if r and not r[0].startswith("timestamp")]
num = lambda s: float(s.split()[0]) if s.split() and s.split()[0][0].isdigit() else 0.0
load = [r for r in rows if num(r[4]) >= 40]
label = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]
if not load:
    print(f"TRACE {label}: no load in {len(rows)} samples")
    sys.exit()
ps = {}
for r in load:
    ps[r[1].strip()] = ps.get(r[1].strip(), 0) + 1
sm, mem, pw, temp = ([num(r[i]) for r in load] for i in (2, 3, 4, 6))
share = lambda i: 100 * sum("Active" in r[i] and "Not" not in r[i] for r in load) / len(load)
print(f"TRACE {label}: {len(load)} load samples | P-state {ps} | SM median {st.median(sm):.0f} low {min(sm):.0f} high {max(sm):.0f} MHz | "
      f"memory median {st.median(mem):.0f} high {max(mem):.0f} MHz | power median {st.median(pw):.0f} max {max(pw):.0f} W | "
      f"temp max {max(temp):.0f} | sw power cap {share(7):.0f}% sw thermal {share(8):.0f}% hw thermal {share(9):.0f}% brake {share(10):.0f}%")
