#!/usr/bin/env python3
"""visiontest: image questions with known answers against an OpenAI-compatible server started with
--vision, using NInferno's own example images (examples/cli/media). Prints per question: pass/fail,
prompt tokens (the image token cost), time to first answer and the answer; appends a summary row."""
import argparse, base64, json, os, time, urllib.request

MEDIA = "/opt/NInferno/examples/cli/media"
# (images, question, accepted answers: any of these substrings, case-insensitive)
CASES = [
    (["visual_chart.png"], "What number is in the title of this image? Reply with the number only.", ["731"]),
    (["visual_chart.png"], "How many red circles are in this image? Reply with a single digit.", ["3", "three"]),
    (["visual_chart.png"], "Which shape is to the right of the blue square? Reply with the shape name only.", ["triangle"]),
    (["natural_scene.png"], "What number is on the sign in this image? Reply with the number only.", ["24"]),
    (["natural_scene.png"], "What color is the roof of the house? Reply with one word.", ["red"]),
    (["compare_left.png", "compare_right.png"],
     "Which shape appears in the second image but not in the first? Reply with the shape name only.", ["star"]),
    (["compare_left.png", "compare_right.png"],
     "How many red circles does the second image have more than the first? Reply with a single digit.", ["1", "one"]),
]


def data_uri(name):
    raw = open(os.path.join(MEDIA, name), "rb").read()
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def ask(base, images, question):
    content = [{"type": "image_url", "image_url": {"url": data_uri(n)}} for n in images]
    content.append({"type": "text", "text": question})
    body = {"model": "x", "messages": [{"role": "user", "content": content}], "max_tokens": 32,
            "temperature": 0}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    return (out["choices"][0]["message"].get("content") or "").strip(), out.get("timings") or {}, \
        out.get("usage") or {}, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="/root/gbench/vision.jsonl")
    a = ap.parse_args()
    rows = []
    for images, q, ok in CASES:
        ans, tm, usage, wall = ask(a.base, images, q)
        hit = any(x.lower() in ans.lower() for x in ok)
        r = {"images": images, "q": q[:60], "ok": hit, "answer": ans[:60],
             "prompt_tokens": usage.get("prompt_tokens"), "prompt_ms": round(tm.get("prompt_ms") or 0),
             "wall": round(wall, 2)}
        rows.append(r)
        print(json.dumps(r), flush=True)
    acc = sum(r["ok"] for r in rows)
    summary = {"label": a.label, "acc": f"{acc}/{len(rows)}", "rows": rows}
    with open(a.out, "a") as f:
        f.write(json.dumps(summary) + "\n")
    print("VISION", a.label, f"{acc}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
