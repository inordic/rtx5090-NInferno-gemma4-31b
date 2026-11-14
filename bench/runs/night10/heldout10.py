"""heldout10.py LABEL LOG: the drafter A/B set, sent to the server on 8080 at each prompt's own temperature (max 512
tokens), with decode speed and MTP acceptance read from the server log. Nothing here was trained on:
  - night 9's held-out prompts, 16 per category (as night 9's heldout.py);
  - night 10's A/B prompts (n10/ab.jsonl: code, maths, English chat, first turns of agent episodes), never collected;
  - every turn of night 10's held-out agent episodes (later turns, tool results in context).
Summaries: all, greedy, sampled, per category, and English (every category but Swedish and oasst2) against the rest.
  -> n10/heldout-LABEL.json and one line per summary"""
import hashlib, json, re, sys, time, urllib.request
from collections import defaultdict

label, logf = sys.argv[1], sys.argv[2]
NOT_ENGLISH = ("swedish", "oasst2")


def is_val(rid):
    return int(hashlib.md5(rid.split("~")[0].encode()).hexdigest(), 16) % 20 == 0


def server_msg(m):
    if m.get("tool_calls"):
        return {"role": "assistant", "content": m.get("content") or "", "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["function"]["name"],
             "arguments": c["function"]["arguments"] if isinstance(c["function"]["arguments"], str) else json.dumps(c["function"]["arguments"], ensure_ascii=False)}}
            for c in m["tool_calls"]]}
    return m


pick = []
by = defaultdict(list)
for l in open("/opt/models/n9/jobs.jsonl"):
    j = json.loads(l)
    if is_val(j["id"]):
        by[j["cat"]].append(j)
for cat in sorted(by):
    pick += by[cat][:16]
pick += [json.loads(l) for l in open("/opt/models/n10/ab.jsonl")]
later = []
for l in open("/opt/models/n10/gen.jsonl"):
    try:
        r = json.loads(l)
    except ValueError:
        continue
    if r["cat"] == "agent" and is_val(r["id"]) and r.get("turn", 0) > 0:
        later.append({**r, "cat": "agent-later"})
pick += later[:24]

if label == "DRY":
    from collections import Counter
    print(len(pick), Counter(j["cat"] for j in pick)); sys.exit(0)
start_line = sum(1 for _ in open(logf))
cats = []
for j in pick:
    body = {"model": "x", "messages": [server_msg(m) for m in j["messages"]], "max_tokens": min(512, j["max_tokens"]), "temperature": j["temperature"]}
    if j["temperature"] > 0:
        body["top_p"] = 0.95
    if j.get("tools"):
        body["tools"] = j["tools"]
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            json.load(r)
        cats.append((j["cat"], j["temperature"], j["id"]))
    except Exception as e:
        print("ERR", j["id"], str(e)[:160], flush=True)
        cats.append((j["cat"], None, j["id"]))
time.sleep(1)
done = [l for i, l in enumerate(open(logf)) if i >= start_line and " done | " in l]
pat_acc = re.compile(r"mtp accepted ([\d,]+)/([\d,]+)")
pat_dec = re.compile(r"output ([\d,]+) .*? decode ([\d.]+) tok/s")
rows = []
ok = [c for c in cats if c[1] is not None]
if len(done) != len(ok):
    print(f"WARNING: {len(ok)} answered requests but {len(done)} done lines in the log", flush=True)
for (cat, temp, rid), l in zip(ok, done):
    a = pat_acc.search(l); d = pat_dec.search(l)
    if not (a and d):
        continue
    out = int(d.group(1).replace(",", ""))
    rows.append({"id": rid, "cat": cat, "temp": temp, "out": out, "decode": float(d.group(2)),
                 "acc": int(a.group(1).replace(",", "")), "drafted": int(a.group(2).replace(",", ""))})


def summary(rs):
    out = sum(r["out"] for r in rs)
    secs = sum(r["out"] / r["decode"] for r in rs if r["decode"] > 0)
    return {"requests": len(rs), "tokens": out, "decode_tok_s": round(out / secs, 1) if secs else 0,
            "accepted": round(100 * sum(r["acc"] for r in rs) / max(1, sum(r["drafted"] for r in rs)), 1)}


res = {"all": summary(rows), "greedy": summary([r for r in rows if r["temp"] == 0.0]),
       "sampled": summary([r for r in rows if r["temp"] > 0]),
       "english": summary([r for r in rows if r["cat"] not in NOT_ENGLISH]),
       "other-languages": summary([r for r in rows if r["cat"] in NOT_ENGLISH])}
for cat in sorted({r["cat"] for r in rows}):
    res[cat] = summary([r for r in rows if r["cat"] == cat])
json.dump({"label": label, "summary": res, "rows": rows, "failed": [c[2] for c in cats if c[1] is None]},
          open(f"/root/gbench/n10/heldout-{label}.json", "w"), indent=1)
for k, v in res.items():
    print(f"HELDOUT {label} {k}: {v}")
