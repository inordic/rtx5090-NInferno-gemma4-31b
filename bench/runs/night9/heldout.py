"""heldout.py LABEL LOG: send the held-out prompts (never trained on) to the server on 8080 at their own
temperatures, then read the server log for their decode speed and MTP acceptance.
  -> n9/heldout-LABEL.json and one summary line per category"""
import hashlib, json, re, sys, time, urllib.request
from collections import defaultdict

label, logf = sys.argv[1], sys.argv[2]


def is_val(rid):
    return int(hashlib.md5(rid.encode()).hexdigest(), 16) % 20 == 0


jobs = [json.loads(l) for l in open("/opt/models/n9/jobs.jsonl")]
by = defaultdict(list)
for j in jobs:
    if is_val(j["id"]):
        by[j["cat"]].append(j)
pick = []
for cat in sorted(by):
    pick += by[cat][:16]
start_line = sum(1 for _ in open(logf))
cats = []
for j in pick:
    body = {"model": "x", "messages": j["messages"], "max_tokens": min(512, j["max_tokens"]), "temperature": j["temperature"]}
    if j["temperature"] > 0:
        body["top_p"] = 0.95
    if j.get("tools"):
        body["tools"] = j["tools"]
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            json.load(r)
        cats.append((j["cat"], j["temperature"]))
    except Exception as e:
        cats.append((j["cat"], None))
time.sleep(1)
done = [l for i, l in enumerate(open(logf)) if i >= start_line and " done | " in l]
pat_acc = re.compile(r"mtp accepted ([\d,]+)/([\d,]+)")
pat_dec = re.compile(r"output ([\d,]+) .*? decode ([\d.]+) tok/s")
rows = []
for (cat, temp), l in zip([c for c in cats if c[1] is not None], done):
    a = pat_acc.search(l); d = pat_dec.search(l)
    if not (a and d):
        continue
    out = int(d.group(1).replace(",", ""))
    rows.append({"cat": cat, "temp": temp, "out": out, "decode": float(d.group(2)),
                 "acc": int(a.group(1).replace(",", "")), "drafted": int(a.group(2).replace(",", ""))})


def summary(rs):
    out = sum(r["out"] for r in rs)
    secs = sum(r["out"] / r["decode"] for r in rs if r["decode"] > 0)
    return {"requests": len(rs), "tokens": out, "decode_tok_s": round(out / secs, 1) if secs else 0,
            "accepted": round(100 * sum(r["acc"] for r in rs) / max(1, sum(r["drafted"] for r in rs)), 1)}


res = {"all": summary(rows), "greedy": summary([r for r in rows if r["temp"] == 0.0]),
       "sampled": summary([r for r in rows if r["temp"] > 0])}
for cat in sorted({r["cat"] for r in rows}):
    res[cat] = summary([r for r in rows if r["cat"] == cat])
json.dump({"label": label, "summary": res, "rows": rows}, open(f"/root/gbench/n9/heldout-{label}.json", "w"), indent=1)
for k, v in res.items():
    print(f"HELDOUT {label} {k}: {v}")
