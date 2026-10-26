#!/usr/bin/env python3
"""needle: long-context retrieval + prefill/decode speed at given context sizes.

For each target length L and depth d, builds a document of ~L tokens from the Federal Register
haystack, inserts three key/value needles at depths d, (d+0.33)%1, (d+0.66)%1, and asks for one of
them. Reports exact-match accuracy, prompt tokens, prefill tok/s and decode tok/s.
"""
import argparse, json, random, time, urllib.request

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
            keys = [f"{rnd.choice(['amber','cobalt','violet','saffron','teal'])}-{rnd.choice(['falcon','otter','lynx','heron','badger'])}-{rnd.randrange(1000, 9999)}" for _ in range(3)]
            names = ["Project Kestrel", "Project Juniper", "Project Halcyon"]
            depths = [d, (d + 0.33) % 1.0, (d + 0.66) % 1.0]
            pieces = sorted(zip(depths, names, keys))
            out_doc, last = [], 0
            for dep, nm, k in pieces:
                cut = int(dep * len(doc))
                cut = doc.rfind("\n", last, cut) if doc.rfind("\n", last, cut) > last else cut
                out_doc.append(doc[last:cut])
                out_doc.append(f"\nNOTE: The access code for {nm} is {k}.\n")
                last = cut
            out_doc.append(doc[last:])
            target = 0  # ask for the needle inserted at depth d
            nm, k = names[target], keys[target]
            prompt = (f"[run {time.time_ns()}]\n<document>\n" + "".join(out_doc) +
                      f"\n</document>\n\nWhat is the access code for {nm}? Reply with the code only.")
            ans, tm, wall = ask(base, prompt)
            ok = k in ans
            r = {"L": L, "depth": d, "ok": ok, "ptok": tm.get("prompt_n"),
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


if __name__ == "__main__":
    main()
