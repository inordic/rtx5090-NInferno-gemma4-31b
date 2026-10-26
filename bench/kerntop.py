#!/usr/bin/env python3
"""Top kernels of an nsys cuda_gpu_kern_sum CSV, grouped by kernel family."""
import csv, re, sys

f = sys.argv[1]
rows = list(csv.DictReader(open(f)))
key = "Total Time (ns)"
tot = sum(float(r[key]) for r in rows)
print(f"total kernel time {tot / 1e6:.1f} ms over {sum(int(r['Instances']) for r in rows)} launches")
fam = {}
for r in rows:
    name = r["Name"]
    m = re.match(r"(?:void )?(?:[\w:]+::)*([A-Za-z_][\w]*)", name)
    k = m.group(1) if m else name[:40]
    fam[k] = fam.get(k, 0.0) + float(r[key])
print("-- by kernel family")
for k, t in sorted(fam.items(), key=lambda kv: -kv[1])[:16]:
    print(f"{100 * t / tot:5.1f}%  {t / 1e6:8.1f} ms  {k}")
print("-- top instantiations")
for r in sorted(rows, key=lambda r: -float(r[key]))[:14]:
    t = float(r[key])
    print(f"{100 * t / tot:5.1f}%  {t / 1e6:8.1f} ms  n={int(r['Instances']):5d}  {r['Name'][:150]}")
