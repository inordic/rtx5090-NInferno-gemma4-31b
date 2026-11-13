"""voice: sampled replies (temperature 0.9, top_p 0.95) from the companion persona, 3 per prompt, no tools.
  python3 voice.py LABEL  -> n8/voice/LABEL.json"""
import json, os, sys, time, urllib.request
sys.path.insert(0, "/root/gbench/n8")
SYSTEM = open("/root/gbench/n8/companionfit.py").read().split('SYSTEM = (')[1].split(')\nTOOLS')[0]
SYSTEM = eval("(" + SYSTEM + ")") + " Today is Friday 13 November 2026, 19:30 local time."
PROMPTS = [
    ("casual_en", "hey ava, long day at work. my boss moved the deadline again.", ""),
    ("feelings", "how are you feeling today, honestly?", "Your current state: mood a bit lonely, energy low; Alex last talked to you three days ago."),
    ("casual_sv", "hej ava! hur mår du idag? jag är helt slut efter veckan.", ""),
    ("anywhere", "if you could go anywhere tonight, where would you go?", ""),
    ("thinking", "tell me something you've been thinking about lately", "Your current state: mood curious, energy high."),
    ("review", "ugh. my code review got torn apart today", ""),
]
def ask(msgs):
    body = {"model": "x", "messages": msgs, "max_tokens": 250, "temperature": 0.9, "top_p": 0.95}
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"].get("content") or ""
out = {}
for k, u, ex in PROMPTS:
    msgs = [{"role": "system", "content": SYSTEM + (" " + ex if ex else "")}, {"role": "user", "content": u}]
    out[k] = [ask(msgs) for _ in range(3)]
os.makedirs("/root/gbench/n8/voice", exist_ok=True)
json.dump(out, open(f"/root/gbench/n8/voice/{sys.argv[1]}.json", "w"), ensure_ascii=False, indent=1)
print(f"VOICE {sys.argv[1]}: {sum(len(v) for v in out.values())} samples")
