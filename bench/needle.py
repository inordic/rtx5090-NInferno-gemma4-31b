#!/usr/bin/env python3
"""needle: long-context retrieval + prefill/decode speed at given context sizes.

For each target length L and depth d, builds a document of ~L characters x cpt from the Federal Register
haystack, inserts three key/value needles at depths d, (d+0.33)%1, (d+0.66)%1, and asks for one of
them. Reports accuracy, the actual prompt tokens, prefill tok/s and decode tok/s.
Scoring: a row passes only if the answer contains the wanted code and no other code of the same shape (asking
for one code and getting all three is a miss); "exact" also records whether the answer is the code alone.
--distractors N adds N near-miss notes (a revoked old code for the asked project, codes for similar names).
--multihop asks in two steps: the asked project's note names an alias, and only the alias's note holds the code.
Exit code 1 if any row fails.
"""
import argparse, json, random, re, sys, time, urllib.request

CODE = re.compile(r"\b[a-z]+-[a-z]+-\d{4}\b")

HAY = "/opt/NInferno/examples/cli/messages/long_niah_256k.json"


def blob():
    user = json.load(open(HAY))[-1]["content"]
    return user.split("<document>\n", 1)[1].split("\n</document>", 1)[0]


def ask(base, prompt, max_tokens=48):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
        out = json.load(r)
    tm = out.get("timings") or {}
    return out["choices"][0]["message"].get("content") or "", tm, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--label", required=True)
    ap.add_argument("--lengths", default="16000,64000")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--cpt", type=float, default=4.06)
    ap.add_argument("--distractors", type=int, default=0)
    ap.add_argument("--multihop", action="store_true")
    ap.add_argument("--out", default="/root/gbench/needle.jsonl")
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    text = blob()
    while len(text) < 1_400_000:
        text = text + "\n\n" + text
    rnd = random.Random(42)
    rows = []
    for L in [int(x) for x in a.lengths.split(",")]:
        for d in [float(x) for x in a.depths.split(",")]:
            nchar = int(L * a.cpt)
            off = rnd.randrange(0, len(text) - nchar - 1)
            doc = text[off:off + nchar]
            code = lambda: f"{rnd.choice(['amber','cobalt','violet','saffron','teal'])}-{rnd.choice(['falcon','otter','lynx','heron','badger'])}-{rnd.randrange(1000, 9999)}"
            keys = [code() for _ in range(3)]
            names = ["Project Kestrel", "Project Juniper", "Project Halcyon"]
            depths = [d, (d + 0.33) % 1.0, (d + 0.66) % 1.0]
            notes = [f"NOTE: The access code for {nm} is {k}." for nm, k in zip(names, keys)]
            if a.multihop:   # the asked project's note only names an alias; the alias's note, elsewhere, holds the code
                notes[0] = f"NOTE: {names[0]} uses the access code filed under the alias Bluewater-7."
                notes.append(f"NOTE: The access code filed under the alias Bluewater-7 is {keys[0]}.")
                depths.append((d + 0.5) % 1.0)
            for j in range(a.distractors):   # near misses: a revoked old code, and codes for look-alike names
                notes.append(f"NOTE: The old access code for {names[0]} was {code()}; it was revoked and is no longer valid."
                             if j % 2 == 0 else f"NOTE: The access code for {names[0]} II is {code()}.")
                depths.append(rnd.random())
            pieces = sorted(zip(depths, range(len(notes)), notes))
            out_doc, last = [], 0
            for dep, _, note in pieces:
                cut = int(dep * len(doc))
                cut = doc.rfind("\n", last, cut) if doc.rfind("\n", last, cut) > last else cut
                out_doc.append(doc[last:cut])
                out_doc.append(f"\n{note}\n")
                last = cut
            out_doc.append(doc[last:])
            nm, k = names[0], keys[0]   # ask for the needle inserted at depth d
            prompt = (f"[run {time.time_ns()}]\n<document>\n" + "".join(out_doc) +
                      f"\n</document>\n\nWhat is the {'current ' if a.distractors else ''}access code for {nm}? Reply with the code only.")
            ans, tm, wall = ask(base, prompt)
            found = set(CODE.findall(ans.lower()))
            ok = k in found and len(found) == 1
            exact = ans.strip().strip("`'\".").strip() == k
            r = {"L": L, "depth": d, "ok": ok, "exact": exact, "ptok": tm.get("prompt_n"),
                 "pp": round(tm.get("prompt_per_second") or 0, 1),
                 "tg": round(tm.get("predicted_per_second") or 0, 1), "wall": round(wall, 1),
                 "ans": ans.strip()[:60], "want": k}
            rows.append(r)
            print(json.dumps(r), flush=True)
    acc = sum(r["ok"] for r in rows)
    summary = {"label": a.label, "acc": f"{acc}/{len(rows)}", "rows": rows}
    with open(a.out, "a") as f:
        f.write(json.dumps(summary) + "\n")
    print("NEEDLE", a.label, f"{acc}/{len(rows)}", flush=True)
    sys.exit(0 if acc == len(rows) else 1)


if __name__ == "__main__":
    main()
