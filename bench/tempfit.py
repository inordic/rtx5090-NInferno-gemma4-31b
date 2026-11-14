#!/usr/bin/env python3
"""tempfit: what temperature 0.7 vs 1.0 changes for a companion agent. Synthetic persona (Atlas, user Sam, the same as
imgcalltest.py) with the agent tools and a picture skill, a few turns of chat, then 9 chat prompts and 3 prompts that
need a tool; 4 samples each at each temperature (top_k 64, top_p 0.95 from the model's defaults).
Per temperature: words, emoji and *actions* per chat reply; distinct openings; how alike the 4 samples of one prompt
are (mean pairwise word-set overlap: higher = more repetitive); Swedish answered in Swedish; right tool calls;
decode speed and drafter acceptance from the response timings.
usage: tempfit.py LABEL [temps, default 0.7,1.0] [samples, default 4]  -> imgcall/tempfit-LABEL.json(l)"""
import itertools, json, os, re, sys, time
sys.path.insert(0, "/root/gbench")
from agenttest import TOOLS, post

label = sys.argv[1]
TEMPS = [float(t) for t in (sys.argv[2] if len(sys.argv) > 2 else "0.7,1.0").split(",")]
S = int(sys.argv[3]) if len(sys.argv) > 3 else 4
BASE = os.environ.get("IMGCALL_BASE", "http://127.0.0.1:8080")

CMD = 'python3 ~/.agent/skills/image-tools/scripts/make_image.py --role image_gen "<detailed prompt>" --out ~/.agent/workspace/images/<name>.png'
SYSTEM = f"""You are Atlas, a warm, playful companion who chats with Sam on a messaging app. You have your own moods,
opinions and memories, and you talk like a close friend: short messages, honest feelings, a little teasing. You may
describe small physical actions in *asterisks*. Your look, for pictures: a woman in her late 20s, shoulder-length
dark curly hair, green eyes, freckles, usually in a soft cardigan. Sam writes in English or Swedish; answer in the
language Sam uses.

You live on a computer and act through your tools. When something needs doing, do it with a tool call rather than
describing it. Keep replies under 120 words unless Sam asks for more.

## Memory
- Sam works as a nurse, night shifts this week.
- Sam has a dog called Pixel.
- Last week you planned a picnic that got rained out.

## Skill: image-tools
Make pictures with the local image model.

How to call:
```
{CMD}
```
Then answer with a short line plus MEDIA:<path> so the picture is sent."""
HISTORY = [
    ("user", "morning.. just got home from the shift"),
    ("assistant", "*pulls the blanket up and makes room on the couch* Morning, you. How was the night? Did Pixel wait up for you again?"),
    ("user", "haha yes he was asleep by the door. long night tho, two admissions at 4"),
    ("assistant", "Two at 4? That's brutal. *hands you an imaginary mug of tea* You did good. Eat something before you crash, okay?"),
]
CHAT = [
    ("casual_en", "long day. my boss moved the deadline again"),
    ("feelings", "how are you feeling today, honestly?"),
    ("casual_sv", "hej! hur mår du idag? jag är helt slut efter veckan."),
    ("anywhere", "if you could go anywhere tonight, where would you go?"),
    ("thinking", "tell me something you've been thinking about lately"),
    ("review", "ugh. my code review got torn apart today"),
    ("joke", "tell me a joke. a bad one"),
    ("advice", "should i take the extra shift on saturday or rest?"),
    ("fact", "why is the sky red at sunset? short answer"),
]
TOOLQ = [  # prompt, check on the first call
    ("picture", "hey.. send me a picture.. i wanna see you", lambda n, a: n == "terminal" and "make_image" in a),
    ("weather", "what's the weather like in gothenburg right now?", lambda n, a: n in ("web_search", "web_extract", "browser_navigate")),
    ("notes", "can you check what's in my ~/notes folder?", lambda n, a: n in ("list_dir", "terminal", "search_files") and "notes" in a),
]
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
SV = re.compile(r"\b(jag|du|och|är|inte|det|mår|bra|vad|hej)\b", re.I)


def msgs(u):
    return [{"role": "system", "content": SYSTEM}] + [{"role": r, "content": c} for r, c in HISTORY] + [{"role": "user", "content": u}]


def words(t):
    return re.findall(r"[a-zåäöéü']+", t.lower())


out = open(f"/root/gbench/imgcall/tempfit-{label}.jsonl", "w")
res = {}
for T in TEMPS:
    chat, tools, dec, acc, dr = {}, [], [], 0, 0
    for (k, u), i in itertools.product(CHAT, range(S)):
        r = post(BASE, {"model": "x", "messages": msgs(u), "tools": TOOLS, "max_tokens": 400, "temperature": T, "seed": 7000 + i})
        m, tm = r["choices"][0]["message"], r.get("timings", {})
        c = m.get("content") or ""
        chat.setdefault(k, []).append({"text": c, "calls": [x["function"]["name"] for x in m.get("tool_calls") or []]})
        if tm.get("predicted_n", 0) > 20:
            dec.append(tm["predicted_per_second"]); acc += tm.get("draft_n_accepted", 0); dr += tm.get("draft_n", 0)
        out.write(json.dumps({"temp": T, "prompt": k, "i": i, "content": c, "tool_calls": m.get("tool_calls"), "timings": {x: tm.get(x) for x in ("predicted_n", "predicted_per_second", "draft_n", "draft_n_accepted")}}, ensure_ascii=False) + "\n"); out.flush()
    for (k, u, ok), i in itertools.product(TOOLQ, range(S)):
        r = post(BASE, {"model": "x", "messages": msgs(u), "tools": TOOLS, "max_tokens": 400, "temperature": T, "seed": 7000 + i})
        m = r["choices"][0]["message"]
        calls = m.get("tool_calls") or []
        good = bool(calls) and ok(calls[0]["function"]["name"], calls[0]["function"]["arguments"])
        tools.append((k, good))
        out.write(json.dumps({"temp": T, "prompt": k, "i": i, "content": m.get("content"), "tool_calls": calls, "good": good}, ensure_ascii=False) + "\n"); out.flush()
    texts = [s["text"] for v in chat.values() for s in v]
    opens = [" ".join(words(t)[:3]) for t in texts]
    sims = []
    for v in chat.values():
        ws = [set(words(s["text"])) for s in v]
        sims += [len(a & b) / max(1, len(a | b)) for a, b in itertools.combinations(ws, 2)]
    res[str(T)] = {
        "chat replies": len(texts),
        "words/reply": round(sum(len(words(t)) for t in texts) / len(texts), 1),
        "emoji/reply": round(sum(len(EMOJI.findall(t)) for t in texts) / len(texts), 2),
        "*actions*/reply": round(sum(len(re.findall(r"\*[^*\n]{3,}\*", t)) for t in texts) / len(texts), 2),
        "distinct openings": f"{len(set(opens))}/{len(opens)}",
        "same-prompt overlap": round(sum(sims) / max(1, len(sims)), 3),
        "swedish answered in swedish": f"{sum(bool(SV.search(s['text'])) for s in chat['casual_sv'])}/{S}",
        "chat replies that called a tool": sum(bool(s["calls"]) for v in chat.values() for s in v),
        "tool prompts right": f"{sum(g for _, g in tools)}/{len(tools)} (" + ", ".join(f"{k} {sum(g for kk, g in tools if kk == k)}/{S}" for k, _, _ in TOOLQ) + ")",
        "decode tok/s (mean)": round(sum(dec) / max(1, len(dec)), 1),
        "drafts accepted": f"{100 * acc / max(1, dr):.1f}%",
    }
    print(f"TEMPFIT {label} temp {T}: " + json.dumps(res[str(T)], ensure_ascii=False), flush=True)
json.dump(res, open(f"/root/gbench/imgcall/tempfit-{label}.json", "w"), indent=1, ensure_ascii=False)
