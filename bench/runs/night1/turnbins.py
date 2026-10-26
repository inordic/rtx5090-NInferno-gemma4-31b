import csv, re, collections
rows = []
for r in csv.DictReader(open("/root/gbench/prof_turn_cuda_gpu_trace.csv")):
    if r.get("Start (ns)") and r.get("Duration (ns)"):
        rows.append((int(r["Start (ns)"]), int(float(r["Duration (ns)"])), r["Name"]))
rows.sort()
t0 = rows[0][0]
end = max(s + d for s, d, n in rows)
def fam(n):
    if n.startswith("[CUDA memcpy"): return n.replace("[CUDA memcpy ", "cp:").rstrip("]")
    m = re.match(r"(?:void )?(?:[\w:]+::)*([A-Za-z_][\w]*)", n); return m.group(1) if m else n[:30]
lo = end - int(2.2e9)
bins = collections.defaultdict(collections.Counter)
for s, d, n in rows:
    if s >= lo: bins[(s - lo) // int(50e6)][fam(n)] += d
for b in sorted(bins):
    c = bins[b]; tot = sum(c.values())
    print(f"{b*50:5d} ms busy {tot/1e6:5.1f} | " + ", ".join(f"{k}:{v/1e6:.1f}" for k, v in c.most_common(4)))
