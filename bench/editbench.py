#!/usr/bin/env python3
"""editbench: code-editing decode speed - the model returns a ~150-line source file with one small
change, so most output tokens repeat the prompt (where n-gram copy rounds pay). Greedy, three edits;
prints decode tok/s per edit and whether the edit was made, and appends a summary row."""
import argparse, json, statistics, time, urllib.request

EDITS = [
    ("Rename the function `med` to `median` everywhere it appears.", "def median("),
    ("Change the default of `--reps` from 2 to 3.", 'default=3'),
    ("Add a one-line docstring to the function `post` saying it posts a chat request.", 'def post('),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--label", required=True)
    ap.add_argument("--file", default="/root/gbench/gbench.py")
    ap.add_argument("--out", default="/root/gbench/edit.jsonl")
    a = ap.parse_args()
    src = open(a.file).read()
    rows = []
    for instr, check in EDITS:
        prompt = (f"{instr} Return the complete file and nothing else, no explanations.\n\n"
                  f"```python\n{src}```")
        body = {"model": "x", "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096, "temperature": 0}
        req = urllib.request.Request(a.base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t = time.time()
        with urllib.request.urlopen(req, timeout=1800) as r:
            out = json.load(r)
        text = out["choices"][0]["message"].get("content") or ""
        tm = out.get("timings") or {}
        rows.append({"edit": instr[:40], "ok": check in text, "out_tokens": tm.get("predicted_n"),
                     "decode": round(tm.get("predicted_per_second") or 0, 1), "wall": round(time.time() - t, 1)})
        print(json.dumps(rows[-1]), flush=True)
    med = statistics.median(r["decode"] for r in rows)
    with open(a.out, "a") as f:
        f.write(json.dumps({"label": a.label, "decode_median": med, "rows": rows}) + "\n")
    print("EDIT", a.label, "decode median", med, flush=True)


if __name__ == "__main__":
    main()
