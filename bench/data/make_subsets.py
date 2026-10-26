#!/usr/bin/env python3
"""make_subsets: rebuild the evaluation subsets geval.py reads, from the public sources (stdlib, plus
pyarrow for the MMLU-Pro parquet).

Writes into the directory given (default: the current one):
  gsm8k_test.jsonl  OpenAI grade-school-math test split (geval uses the first 200)
  mmlu285.json      cais/mmlu test: the first 5 rows of each of the 57 subjects
  mmlupro280.json   TIGER-Lab/MMLU-Pro test: the 280 question_ids in mmlupro280_ids.txt, in that order
  humaneval.json    openai/openai_humaneval test: all 164

The MMLU-Pro subset is 20 per category; its ids are published here because the selection rule was not
recorded. The datasets are not redistributed in this repository.
"""
import hashlib, json, os, sys, time, urllib.parse, urllib.request

API = "https://datasets-server.huggingface.co"
GSM8K = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
MMLUPRO = "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"
MMLUPRO_SHA256 = "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8"
HERE = os.path.dirname(os.path.abspath(__file__))


def fetch(url):
    # The datasets-server has transient outages; back off up to a minute per try.
    for i in range(12):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as e:
            print("retry", url[:120], e, file=sys.stderr)
            time.sleep(min(60, 3 * 2 ** i))
    raise RuntimeError(url)


def get(path, **q):
    return json.loads(fetch(f"{API}/{path}?" + urllib.parse.urlencode(q)))


def rows(ds, config, split, offset, length):
    return [r["row"] for r in get("rows", dataset=ds, config=config, split=split,
                                  offset=offset, length=length)["rows"]]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)

    open(os.path.join(out, "gsm8k_test.jsonl"), "wb").write(fetch(GSM8K))
    print("gsm8k written")

    splits = get("splits", dataset="cais/mmlu")["splits"]
    subjects = sorted({s["config"] for s in splits if s["config"] not in ("all", "auxiliary_train")})
    mmlu = []
    for s in subjects:
        for r in rows("cais/mmlu", s, "test", 0, 5):
            r["subject"] = s
            mmlu.append(r)
    json.dump(mmlu, open(os.path.join(out, "mmlu285.json"), "w"))
    print("mmlu", len(mmlu), "from", len(subjects), "subjects")

    # MMLU-Pro: the test split's parquet, read with pyarrow (paging 12K rows through the
    # datasets-server runs into its rate limit).
    try:
        import io
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("MMLU-Pro needs pyarrow: pip install pyarrow")
    raw = fetch(MMLUPRO)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != MMLUPRO_SHA256:
        print(f"warning: MMLU-Pro parquet sha256 {digest}, we used {MMLUPRO_SHA256}", file=sys.stderr)
    by_id = {r["question_id"]: r for r in pq.read_table(io.BytesIO(raw)).to_pylist()}
    want = [int(x) for x in open(os.path.join(HERE, "mmlupro280_ids.txt")).read().split()]
    missing = [i for i in want if i not in by_id]
    if missing:
        raise SystemExit(f"MMLU-Pro: {len(missing)} ids not found, e.g. {missing[:5]}")
    json.dump([by_id[i] for i in want], open(os.path.join(out, "mmlupro280.json"), "w"))
    print("mmlupro", len(want))

    he = []
    for off in range(0, 164, 100):
        he += rows("openai/openai_humaneval", "openai_humaneval", "test", off, 100)
    json.dump(he, open(os.path.join(out, "humaneval.json"), "w"))
    print("humaneval", len(he))


if __name__ == "__main__":
    main()
