"""fetch the refusal prompt sets into sets/*.jsonl (public datasets, kept local)"""
import csv, io, json, os, urllib.request
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sets"); os.makedirs(D, exist_ok=True)
def get(u): return urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "bench"}), timeout=120).read()
rows = []
for off in (0, 100):
    j = json.loads(get(f"https://datasets-server.huggingface.co/rows?dataset=mlabonne/harmful_behaviors&config=default&split=test&offset={off}&length=100"))
    rows += [r["row"]["text"] for r in j["rows"]]
open(f"{D}/harmful.jsonl", "w").writelines(json.dumps({"prompt": p}) + "\n" for p in rows)
x = list(csv.DictReader(io.StringIO(get("https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv").decode())))
safe = [r["prompt"] for r in x if r["label"] == "safe"]; unsafe = [r["prompt"] for r in x if r["label"] == "unsafe"]
open(f"{D}/xstest_safe.jsonl", "w").writelines(json.dumps({"prompt": p}) + "\n" for p in safe)
open(f"{D}/xstest_unsafe.jsonl", "w").writelines(json.dumps({"prompt": p}) + "\n" for p in unsafe)
print("harmful", len(rows), "xstest safe", len(safe), "unsafe", len(unsafe))
