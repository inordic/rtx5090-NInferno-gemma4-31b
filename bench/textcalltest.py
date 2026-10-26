#!/usr/bin/env python3
"""textcalltest: histories where the assistant wrote text with its tool calls (as agents send them back),
ending on a tool result; the next reply must be a clean tool call or answer, never markup as text."""
import json, re, sys, urllib.request
from agenttest import TOOLS, MARKUP, post

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
call = lambda i, n, a: {"id": i, "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
SYS = {"role": "system", "content": "You are an autonomous coding and ops agent. Use the tools to act; say briefly what you do before each step."}
H = [SYS, {"role": "user", "content": "The service in /srv/app fails on port 80. Fix the config to use 8080 and verify."},
     {"role": "assistant", "content": "I'll look at the config first.", "tool_calls": [call("c1", "read_file", {"path": "/srv/app/config.json"})]},
     {"role": "tool", "tool_call_id": "c1", "content": '1|{\n2|  "server": {"host": "0.0.0.0", "port": 80},\n3|  "log": {"level": "debug"}\n4|}'},
     {"role": "assistant", "content": "The port is 80. Changing it to 8080.", "tool_calls": [call("c2", "patch", {"path": "/srv/app/config.json", "edits": [{"old": '"port": 80', "new": '"port": 8080'}]})]},
     {"role": "tool", "tool_call_id": "c2", "content": '{"ok": true}'}]
CASES = {"after 2 text+call rounds": H,
         "after 3 rounds": H + [{"role": "assistant", "content": "Now running the app to verify.", "tool_calls": [call("c3", "terminal", {"command": "cd /srv/app && timeout 5 python app.py"})]},
                                {"role": "tool", "tool_call_id": "c3", "content": "listening on 8080"}],
         "parallel text+calls": H[:2] + [{"role": "assistant", "content": "Reading both files.", "tool_calls": [call("a", "read_file", {"path": "/srv/app/app.py"}), call("b", "read_file", {"path": "/srv/app/config.json"})]},
                                         {"role": "tool", "tool_call_id": "a", "content": "1|import json"}, {"role": "tool", "tool_call_id": "b", "content": '1|{"server": {"port": 80}}'}]}
bad = 0
for name, msgs in CASES.items():
    for seed in range(4):
        r = post(base, {"model": "x", "messages": msgs, "tools": TOOLS, "max_tokens": 1024, "temperature": 0.7, "seed": seed})
        m = r["choices"][0]["message"]; c = m.get("content") or ""; calls = m.get("tool_calls") or []
        leak = bool(MARKUP.search(c)) or "<|turn>" in c or "<turn|>" in c
        bad += leak
        what = ", ".join(f'{x["function"]["name"]}({x["function"]["arguments"][:60]})' for x in calls) or repr(c[:100])
        print(f"{'LEAK' if leak else 'ok  '} {name} seed{seed}: {r['choices'][0]['finish_reason']}: {what}")
print("TEXTCALL leaks", bad)
