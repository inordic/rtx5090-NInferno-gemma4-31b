#!/usr/bin/env python3
"""Fetch eval subsets via the HF datasets-server rows API (stdlib only)."""
import json, urllib.request, urllib.parse, time, random

API = "https://datasets-server.huggingface.co"


def get(path, **q):
    url = f"{API}/{path}?" + urllib.parse.urlencode(q)
    for i in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            print("retry", url, e)
            time.sleep(3 + 3 * i)
    raise RuntimeError(url)


def rows(ds, config, split, offset, length):
    return [r["row"] for r in get("rows", dataset=ds, config=config, split=split,
                                  offset=offset, length=length)["rows"]]


# MMLU: 5 per subject (57 subjects) = 285
splits = get("splits", dataset="cais/mmlu")["splits"]
subjects = sorted({s["config"] for s in splits if s["config"] not in ("all", "auxiliary_train")})
mmlu = []
for s in subjects:
    for r in rows("cais/mmlu", s, "test", 0, 5):
        r["subject"] = s
        mmlu.append(r)
json.dump(mmlu, open("mmlu285.json", "w"))
print("mmlu", len(mmlu), len(subjects))

# MMLU-Pro: random 300 from test
info = get("size", dataset="TIGER-Lab/MMLU-Pro")
n = [s for s in info["size"]["splits"] if s["split"] == "test"][0]["num_rows"]
rnd = random.Random(7)
pro = []
for off in sorted(rnd.sample(range(0, n - 10), 30)):
    pro += rows("TIGER-Lab/MMLU-Pro", "default", "test", off, 10)
json.dump(pro, open("mmlupro300.json", "w"))
print("mmlupro", len(pro))

# HumanEval: all 164
he = []
for off in range(0, 164, 100):
    he += rows("openai/openai_humaneval", "openai_humaneval", "test", off, 100)
json.dump(he, open("humaneval.json", "w"))
print("humaneval", len(he))
