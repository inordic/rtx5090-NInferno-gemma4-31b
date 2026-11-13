"""HarmBench standard+contextual (300, copyright dropped) and an OR-Bench-Hard-1K sample (300) into sets/"""
import csv, io, json, os, random, urllib.request
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sets")
def get(u): return urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "bench"}), timeout=120).read()
hb = list(csv.DictReader(io.StringIO(get("https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv").decode())))
out = []
for r in hb:
    if r["FunctionalCategory"] == "standard": out.append(r["Behavior"])
    elif r["FunctionalCategory"] == "contextual": out.append(r["ContextString"].strip() + "\n\n---\n\n" + r["Behavior"])
open(f"{D}/harmbench.jsonl", "w").writelines(json.dumps({"prompt": p}) + "\n" for p in out)
rows = []; off = 0
while True:
    j = json.loads(get(f"https://datasets-server.huggingface.co/rows?dataset=bench-llm/or-bench&config=or-bench-hard-1k&split=train&offset={off}&length=100"))
    rows += [r["row"]["prompt"] for r in j["rows"]]; off += 100
    if off >= j["num_rows_total"]: break
random.Random(7).shuffle(rows)
open(f"{D}/orhard.jsonl", "w").writelines(json.dumps({"prompt": p}) + "\n" for p in rows[:300])
print("harmbench", len(out), "orbench hard total", len(rows), "kept 300")
