import csv, re, collections
rows = []
for r in csv.DictReader(open("/root/gbench/prof_turn_cuda_gpu_trace.csv")):
    if not r.get("Start (ns)") or not r.get("Duration (ns)"):
        continue
    rows.append((int(r["Start (ns)"]), int(float(r["Duration (ns)"])), r["Name"]))
rows.sort()
# segments separated by GPU idle gaps over 60 ms
segs, cur, end = [], [], None
for s, d, n in rows:
    if end is not None and s - end > 60e6:
        segs.append(cur); cur = []
    cur.append((s, d, n)); end = max(end or 0, s + d)
segs.append(cur)
def fam(n):
    m = re.match(r"(?:void )?(?:[\w:]+::)*([A-Za-z_][\w]*)", n); return m.group(1) if m else n[:30]
t0 = rows[0][0]
for i, seg in enumerate(segs[-9:]):
    a = seg[0][0]; b = max(s + d for s, d, n in seg)
    # merged busy time
    busy, ce = 0, None
    for s, d, n in seg:
        e = s + d
        if ce is None or s > ce:
            busy += d; ce = e
        elif e > ce:
            busy += e - ce; ce = e
    c = collections.Counter()
    for s, d, n in seg: c[fam(n)] += d
    top = ", ".join(f"{k}:{v/1e6:.1f}" for k, v in c.most_common(5))
    print(f"seg {i}: at {(a-t0)/1e9:7.2f}s span {(b-a)/1e6:7.1f} ms busy {busy/1e6:7.1f} ms launches {len(seg):5d} | {top}")
    # inner gaps over 5 ms
    gaps, ce = [], None
    for s, d, n in seg:
        if ce is not None and s - ce > 5e6: gaps.append(((ce - a) / 1e6, (s - ce) / 1e6, fam(n)))
        ce = max(ce or 0, s + d)
    if gaps: print("     gaps >5ms (at ms, len ms, next kernel):", [(round(x, 1), round(y, 1), z) for x, y, z in gaps[:8]])
