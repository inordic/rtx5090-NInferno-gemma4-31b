#!/usr/bin/env python3
"""geval: quick quality suite against an OpenAI-compatible server (stdlib only).

gsm8k   : first N test problems, CoT, greedy, exact match on the final number
mmlu    : 285 (5 per subject), direct letter answer
mmlupro : 280 (20 per category), direct letter answer (10 options)
humaneval: 164, greedy completion, executed against the tests (subprocess, timeout)
"""
import argparse, json, re, subprocess, sys, tempfile, time, os, urllib.request
from concurrent.futures import ThreadPoolExecutor

D = "/root/gbench/data"


def ask(base, msgs, max_tokens, temperature=0.0):
    body = {"model": "x", "messages": msgs, "max_tokens": max_tokens, "temperature": temperature}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    for i in range(3):
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                out = json.load(r)
            return out["choices"][0]["message"].get("content") or "", out.get("usage", {})
        except Exception as e:
            err = e
            time.sleep(2)
    return f"<<error {err}>>", {}


def num(s):
    s = s.replace(",", "").replace("$", "")
    m = re.findall(r"answer is[:\s]*\**\s*(-?\d+(?:\.\d+)?)", s, re.I)
    if not m:
        m = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    v = float(m[-1])
    return int(v) if v == int(v) else v


def gsm8k(base, n, conc):
    items = [json.loads(l) for l in open(f"{D}/gsm8k_test.jsonl")][:n]

    def one(it):
        q = (it["question"] + "\n\nSolve step by step, then finish with a line of the form "
             "'The answer is: <number>'.")
        out, u = ask(base, [{"role": "user", "content": q}], 1024)
        gold = num(it["answer"].split("####")[-1])
        return {"ok": num(out) == gold, "gold": gold, "pred": num(out), "toks": u.get("completion_tokens", 0)}

    return run(one, items, conc)


LET = "ABCDEFGHIJ"


def letter(s, k):
    s = s.strip()
    m = re.search(r"(?:answer is|Answer:)\s*\(?([A-J])\)?", s)
    if m and m.group(1) in LET[:k]:
        return m.group(1)
    m = re.match(r"\(?([A-J])[\)\.\:\s]", s + " ")
    if m and m.group(1) in LET[:k]:
        return m.group(1)
    m = re.findall(r"\b([A-J])\b", s)
    for x in m:
        if x in LET[:k]:
            return x
    return None


def mcq(base, items, conc, qkey, optkey, goldfn):
    def one(it):
        opts = it[optkey]
        q = it[qkey].strip() + "\n\n" + "\n".join(f"{LET[i]}. {o}" for i, o in enumerate(opts))
        q += "\n\nAnswer with the letter of the correct option only."
        out, u = ask(base, [{"role": "user", "content": q}], 8)
        pred = letter(out, len(opts))
        gold = goldfn(it)
        return {"ok": pred == gold, "gold": gold, "pred": pred}

    return run(one, items, conc)


def humaneval(base, conc):
    items = json.load(open(f"{D}/humaneval.json"))

    def one(it):
        q = ("Complete the following Python function. Return the full function (with imports) in a "
             "single ```python code block, no explanation.\n\n```python\n" + it["prompt"] + "```")
        out, u = ask(base, [{"role": "user", "content": q}], 1024)
        m = re.findall(r"```(?:python)?\n(.*?)```", out, re.S)
        code = m[0] if m else out
        if f"def {it['entry_point']}" not in code:
            code = it["prompt"] + code
        prog = code + "\n\n" + it["test"] + f"\n\ncheck({it['entry_point']})\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir="/tmp") as f:
            f.write(prog)
            path = f.name
        try:
            r = subprocess.run([sys.executable, "-I", path], capture_output=True, timeout=20,
                               cwd="/tmp")
            ok = r.returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
        finally:
            os.unlink(path)
        return {"ok": ok, "task": it["task_id"]}

    return run(one, items, conc)


def run(fn, items, conc):
    t = time.time()
    with ThreadPoolExecutor(conc) as ex:
        res = list(ex.map(fn, items))
    return {"score": sum(r["ok"] for r in res), "n": len(res), "secs": round(time.time() - t),
            "detail": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--label", required=True)
    ap.add_argument("--suites", default="gsm8k,mmlu,mmlupro,humaneval")
    ap.add_argument("--gsm-n", type=int, default=200)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--out", default="/root/gbench/quality.jsonl")
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    row = {"label": a.label, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    details = {}
    for s in a.suites.split(","):
        if s == "gsm8k":
            r = gsm8k(base, a.gsm_n, a.conc)
        elif s == "mmlu":
            r = mcq(base, json.load(open(f"{D}/mmlu285.json")), a.conc, "question", "choices",
                    lambda it: LET[it["answer"]])
        elif s == "mmlupro":
            r = mcq(base, json.load(open(f"{D}/mmlupro280.json")), a.conc, "question", "options",
                    lambda it: it["answer"])
        elif s == "humaneval":
            r = humaneval(base, a.conc)
        else:
            continue
        row[s] = f"{r['score']}/{r['n']}"
        row[s + "_secs"] = r["secs"]
        details[s] = r["detail"]
        print(f"{s}: {r['score']}/{r['n']} ({r['secs']}s)", flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    tag = re.sub(r"[^A-Za-z0-9]+", "_", a.label)[:60]
    json.dump(details, open(f"/root/gbench/qdetail_{tag}.json", "w"))
    print("ROW " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
