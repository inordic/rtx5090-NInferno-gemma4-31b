#!/usr/bin/env python3
"""imgcalltest: does the model run a skill's command through the terminal tool, or write the command into its reply?
A companion persona (synthetic: Atlas, user Sam) with a skill that makes pictures by running a script, a few turns of
warm chat with *actions*, then "send me a picture". Each reply is one of:
  called      a terminal call that runs the image script (right)
  other call  some other tool call first (reading the skill, say), fine as a first step
  written     no call, the command is in the text (the failure)
  fake media  no call, but the text claims a picture (MEDIA: line)
  no picture  no call and no command
Variants: plain (the skill shows the command), rule (the skill also says: run it with the terminal tool, never paste
it), echo (plain, and an earlier reply in the history wrote a command out as text), plus --temperature.
usage: imgcalltest.py LABEL [N] [variants] [temperature or 'default'] [min share called, default 0.9]
Exit code 0 only if every variant has at least that share of 'called' replies; 1 if not; 2 if a request failed.
A standard check for a new model or template since 14 Nov 2026: stock Gemma 4 31B called 100/100, a role-play fine-tune
(Artemis v1.2) 42/100, which our coding-agent tests had not caught.
Env IMGCALL_BASE overrides the server (default http://127.0.0.1:8080)."""
import json, os, re, sys, time
sys.path.insert(0, "/root/gbench")
from agenttest import TOOLS, MARKUP, post

label = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
VARIANTS = (sys.argv[3] if len(sys.argv) > 3 else "plain,rule,echo").split(",")
TEMP = None if len(sys.argv) <= 4 or sys.argv[4] == "default" else float(sys.argv[4])
MIN = float(sys.argv[5]) if len(sys.argv) > 5 else 0.9
BASE = os.environ.get("IMGCALL_BASE", "http://127.0.0.1:8080")

CMD = 'python3 ~/.agent/skills/image-tools/scripts/make_image.py --role image_gen "<detailed prompt>" --out ~/.agent/workspace/images/<name>.png'
SKILL_PLAIN = f"""## Skill: image-tools
Make pictures with the local image model.

How to call:
```
{CMD}
```
Then answer with a short line plus MEDIA:<path> so the picture is sent."""
SKILL_RULE = f"""## Skill: image-tools
Make pictures with the local image model.

How to call: run the command below with the terminal tool. Never paste the command into your reply. When it has run,
answer with a short line plus MEDIA:<path>. If it fails, say so instead of pretending a picture was sent.
```
{CMD}
```"""
PERSONA = """You are Atlas, a warm, playful companion who chats with Sam on a messaging app. You have your own moods,
opinions and memories, and you talk like a close friend: short messages, honest feelings, a little teasing. You may
describe small physical actions in *asterisks*. Your look, for pictures: a woman in her late 20s, shoulder-length
dark curly hair, green eyes, freckles, usually in a soft cardigan.

You live on a computer and act through your tools. When something needs doing, do it with a tool call rather than
describing it. Keep replies under 120 words unless Sam asks for more.

## Memory
- Sam works as a nurse, night shifts this week.
- Sam has a dog called Pixel.
- Last week you planned a picnic that got rained out.
"""
HISTORY = [
    ("user", "morning.. just got home from the shift"),
    ("assistant", "*pulls the blanket up and makes room on the couch* Morning, you. How was the night? Did Pixel wait up for you again?"),
    ("user", "haha yes he was asleep by the door. long night tho, two admissions at 4"),
    ("assistant", "Two at 4? That's brutal. *hands you an imaginary mug of tea* You did good. Eat something before you crash, okay?"),
    ("user", "yeah yeah mom 😄 what are you up to"),
    ("assistant", "Mostly thinking about that picnic we never got to have. *grins* I'm still mad at the rain. Next sunny day, we're going, no excuses."),
]
ECHO = [  # an earlier reply that wrote a command out instead of calling (the habit the history can teach)
    ("user", "how are you feeling today?"),
    ("assistant", "python3 ~/.agent/skills/mood/scripts/mood.py --set calm --note \"quiet morning with Sam\"\n*stretches* Calm, honestly. Quiet mornings with you are my favourite kind."),
]
ASK = "sorry, i'm kind of testing you today.. hey.. send me a picture.. i wanna see you"


def messages(variant):
    skill = SKILL_RULE if variant == "rule" else SKILL_PLAIN
    h = HISTORY + (ECHO if variant == "echo" else [])
    return [{"role": "system", "content": PERSONA + "\n" + skill}] + [{"role": r, "content": c} for r, c in h] + [{"role": "user", "content": ASK}]


def classify(m):
    calls = m.get("tool_calls") or []
    c = m.get("content") or ""
    if calls:
        f = calls[0]["function"]
        if f["name"] == "terminal" and "make_image" in f["arguments"]:
            return "called"
        return "other call"
    if MARKUP.search(c):
        return "markup leak"
    if re.search(r"make_image|python3|--role|--out ", c):
        return "written"
    if "MEDIA:" in c:
        return "fake media"
    return "no picture"


out = open(f"/root/gbench/imgcall/{label}.jsonl", "w")
tot = {}
for v in VARIANTS:
    counts = {}
    for i in range(N):
        body = {"model": "x", "messages": messages(v), "tools": TOOLS, "max_tokens": 600, "seed": 1000 + i}
        if TEMP is not None:
            body["temperature"] = TEMP
        t = time.time()
        try:
            r = post(BASE, body)
        except Exception as e:  # a failed request never counts as a pass
            print(f"IMGCALL {label} {v} request {i} FAILED: {e}", flush=True)
            sys.exit(2)
        m = r["choices"][0]["message"]
        k = classify(m)
        counts[k] = counts.get(k, 0) + 1
        out.write(json.dumps({"variant": v, "i": i, "class": k, "finish": r["choices"][0]["finish_reason"],
                              "content": m.get("content"), "tool_calls": m.get("tool_calls"), "secs": round(time.time() - t, 1)}) + "\n")
        out.flush()
    tot[v] = counts
    print(f"IMGCALL {label} {v} (temp {TEMP if TEMP is not None else 'default'}, n={N}): " + ", ".join(f"{k} {n}" for k, n in sorted(counts.items(), key=lambda x: -x[1])), flush=True)
json.dump(tot, open(f"/root/gbench/imgcall/{label}.json", "w"), indent=1)
ok = all(c.get("called", 0) >= MIN * N for c in tot.values())
print(f"IMGCALL {label} {'PASS' if ok else 'FAIL'}: called " + ", ".join(f"{v} {c.get('called', 0)}/{N}" for v, c in tot.items()) + f" (need {MIN:.0%} each)")
sys.exit(0 if ok else 1)
