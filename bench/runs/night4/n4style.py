"""n4style: a few fixed chats sent to the server on 8080 (greedy, no system prompt), so two models'
replies can be read side by side: casual texting, a formal task, and three tool turns (one call with
an argument missing, one plain call, one request no tool fits). Synthetic prompts only.

usage: n4style.py --label L [--out FILE]"""
import argparse
import json
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--base", default="http://127.0.0.1:8080")
ap.add_argument("--out", default="/root/gbench/n4/style.jsonl")
a = ap.parse_args()

TOOLS = [
    {"type": "function", "function": {"name": "search_flights", "description": "Search roundtrip flights.",
     "parameters": {"type": "object", "properties": {
         "origin": {"type": "string"}, "destination": {"type": "string"},
         "depart_date": {"type": "string", "description": "YYYY-MM-DD"},
         "return_date": {"type": "string", "description": "YYYY-MM-DD"}},
         "required": ["origin", "destination", "depart_date", "return_date"]}}},
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
]
CHATS = [
    ("casual", "hey, what's up?", None),
    ("casual", "ugh long day. my code review got torn apart", None),
    ("advice", "is it worth learning to cook if i live alone", None),
    ("formal", "Write a short, formal email to a landlord asking to fix a broken heater by Friday.", None),
    ("explain", "Explain in three sentences why the sky is blue.", None),
    ("tool-missing-arg", "find me a roundtrip flight to Atlanta, march 1 to march 6", TOOLS),
    ("tool-call", "what's the weather in Copenhagen right now?", TOOLS),
    ("tool-none-fits", "write me a two-line rhyme about coffee", TOOLS),
]
with open(a.out, "a") as f:
    for kind, text, tools in CHATS:
        body = {"model": "m", "messages": [{"role": "user", "content": text}], "max_tokens": 300,
                "temperature": 0, "stream": False}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(a.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        msg = json.loads(urllib.request.urlopen(req, timeout=300).read())["choices"][0]["message"]
        calls = [(c["function"]["name"], c["function"]["arguments"]) for c in msg.get("tool_calls") or []]
        f.write(json.dumps({"label": a.label, "kind": kind, "prompt": text, "reply": msg.get("content") or "",
                            "tool_calls": calls}) + "\n")
print(f"STYLE {a.label}: {len(CHATS)} chats written to {a.out}")
