import csv, re, collections
rows = []
for r in csv.DictReader(open("/root/gbench/prof_turn_cuda_gpu_trace.csv")):
    if r.get("Start (ns)") and r.get("Duration (ns)"):
        rows.append((int(r["Start (ns)"]), int(float(r["Duration (ns)"])), r["Name"]))
rows.sort()
t0 = rows[0][0]
rows = [x for x in rows if (x[0] - t0) / 1e9 > 26.0]   # the request phase (seg 7 above)
def fam(n):
    m = re.match(r"(?:void )?(?:[\w:]+::)*([A-Za-z_][\w]*)", n); return m.group(1) if m else n[:30]
segs, cur, end = [], [], None
for s, d, n in rows:
    if end is not None and s - end > 25e6:
        segs.append(cur); cur = []
    cur.append((s, d, n)); end = max(end or 0, s + d)
segs.append(cur)
prev_end = None
for i, seg in enumerate(segs):
    a = seg[0][0]; b = max(s + d for s, d, n in seg)
    busy, ce = 0, None
    for s, d, n in seg:
        e = s + d
        if ce is None or s > ce: busy += d; ce = e
        elif e > ce: busy += e - ce; ce = e
    c = collections.Counter()
    for s, d, n in seg: c[fam(n)] += d
    top = ", ".join(f"{k}:{v/1e6:.1f}" for k, v in c.most_common(6))
    gap = (a - prev_end) / 1e6 if prev_end else 0
    print(f"seg {i}: idle-before {gap:6.1f} ms span {(b-a)/1e6:7.1f} ms busy {busy/1e6:7.1f} ms n {len(seg):5d} | {top}")
    prev_end = b
