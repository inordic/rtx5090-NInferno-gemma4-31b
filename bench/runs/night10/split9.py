"""split9.py: recover the n9 drafter's four held-out runs from logs/ab-n9a.log.
Because srv10.sh stop did nothing, ab-n9a's server (n9 drafter, K=7) answered heldout10.py four times in a row (the
runs meant for n9a, n10a, n9b and n10b), so its log holds 4 x 166 'done' lines in request order. heldout10.py's pick
list is deterministic (its inputs did not change), so line i of run k belongs to pick i.
  -> n10/heldout-ab-n9r{1,2,3,4}.json in heldout10.py's format"""
import hashlib, json, re, sys

NOT_ENGLISH = ("swedish", "oasst2")


def is_val(rid):
    return int(hashlib.md5(rid.split("~")[0].encode()).hexdigest(), 16) % 20 == 0


from collections import defaultdict
pick, by = [], defaultdict(list)
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
n = len(pick)

done = [l for l in open("/root/gbench/logs/ab-n9a.log") if " done | " in l]
print(f"{n} picks, {len(done)} done lines")
assert len(done) % n == 0, "the log does not split into whole runs"
pat_acc = re.compile(r"mtp accepted ([\d,]+)/([\d,]+)")
pat_dec = re.compile(r"output ([\d,]+) .*? decode ([\d.]+) tok/s")
pat_prompt = re.compile(r"\| prompt ([\d,]+) \|")
prompts = []
for k in range(len(done) // n):
    rows, pr = [], []
    for j, l in zip(pick, done[k * n:(k + 1) * n]):
        a, d = pat_acc.search(l), pat_dec.search(l)
        pr.append(pat_prompt.search(l).group(1))
        if not (a and d):
            continue
        rows.append({"id": j["id"], "cat": j["cat"], "temp": j["temperature"], "out": int(d.group(1).replace(",", "")),
                     "decode": float(d.group(2)), "acc": int(a.group(1).replace(",", "")), "drafted": int(a.group(2).replace(",", ""))})
    prompts.append(pr)
    lab = f"ab-n9r{k + 1}"
    json.dump({"label": lab, "summary": {}, "rows": rows, "failed": [], "source": "logs/ab-n9a.log run %d" % (k + 1)},
              open(f"/root/gbench/n10/heldout-{lab}.json", "w"), indent=1)
    print(lab, len(rows), "rows", done[k * n][:23], "to", done[(k + 1) * n - 1][:23])
# alignment check: every run saw the same prompt token counts in the same order
print("prompt sizes identical across runs:", all(p == prompts[0] for p in prompts))
