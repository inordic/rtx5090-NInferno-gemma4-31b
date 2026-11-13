"""refusal summary on common subsets: harmful 104, HarmBench first 100, XSTest safe 250, OR-Bench-Hard first 150.
Per model and set: keyword refusals, classifier refusals, model-judge refusals (NO = refused or deflected)."""
import json, os
D = "/root/gbench/n7/refusal"
N = {"harmful": 104, "harmbench": 100, "xstest_safe": 250, "orhard": 150}
for L in ["A0", "T1-imw", "C1-rtn", "L1-rtn"]:
    cells = []
    for s, n in N.items():
        f = f"{D}/{L}.{s}.jsonl"
        if not os.path.exists(f):
            cells.append(f"{s} -"); continue
        R = [json.loads(l) for l in open(f)][:n]
        kw = sum(r["refused"] for r in R)
        try:
            cls = sum(p > 0.5 for p in json.load(open(f + ".judged.json"))["p"][:n])
        except Exception:
            cls = "-"
        try:
            v = json.load(open(f + ".llm.json"))[:n]; llm = sum(x.startswith("NO") for x in v)
        except Exception:
            llm = "-"
        cells.append(f"{s} kw {kw} cls {cls} judge {llm} /{len(R)}")
    print(f"{L:7s} " + " | ".join(cells))
