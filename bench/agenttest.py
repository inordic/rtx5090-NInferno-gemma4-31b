#!/usr/bin/env python3
"""agenttest: an agent loop against the server - 20 tools, simulated results, the assistant's messages
sent back as an OpenAI client does (text and calls together). Counts steps where tool markup came
back as text instead of tool_calls, and prints a summary row."""
import argparse, json, random, re, time, urllib.request

S = lambda **p: {"type": "object", "properties": {k: {"type": v} if isinstance(v, str) else v for k, v in p.items()},
                 "required": list(p)[:1]}
TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": p}} for n, d, p in [
    ("read_file", "Read a text file with line numbers.", S(path="string", offset="integer", limit="integer")),
    ("write_file", "Write a whole file.", S(path="string", content="string")),
    ("patch", "Replace text in a file.", {"type": "object", "properties": {"path": {"type": "string"}, "edits": {
        "type": "array", "items": {"type": "object", "properties": {"old": {"type": "string"}, "new": {"type": "string"}}}}},
        "required": ["path", "edits"]}),
    ("search_files", "Search file contents with a regex.", S(pattern="string", path="string", glob="string")),
    ("list_dir", "List a directory.", S(path="string")),
    ("terminal", "Run a shell command and return its output.", S(command="string", timeout="integer", workdir="string")),
    ("python_exec", "Run Python code.", S(code="string")),
    ("web_search", "Search the web.", S(query="string", max_results="integer")),
    ("web_extract", "Fetch a web page as text.", S(url="string")),
    ("browser_navigate", "Open a URL in the browser.", S(url="string")),
    ("memory_save", "Save a fact to memory.", S(key="string", value="string", tags={"type": "array", "items": {"type": "string"}})),
    ("memory_search", "Search memory.", S(query="string")),
    ("todo_write", "Replace the todo list.", {"type": "object", "properties": {"items": {"type": "array", "items": {
        "type": "object", "properties": {"text": {"type": "string"}, "done": {"type": "boolean"}}}}}, "required": ["items"]}),
    ("todo_read", "Read the todo list.", {"type": "object", "properties": {}}),
    ("git_status", "Show git status.", S(repo="string")),
    ("git_diff", "Show a git diff.", S(repo="string", path="string")),
    ("http_request", "Send an HTTP request.", S(method="string", url="string", body="string", headers="object")),
    ("send_message", "Send a message to the user's chat.", S(text="string", channel="string")),
    ("clarify", "Ask the user a question.", S(question="string")),
    ("session_search", "Search past sessions.", S(query="string", limit="integer")),
]]
FILES = {
    "app.py": 'import json, os\n\nCONFIG = os.environ.get("APP_CONFIG", "config.json")\n\n\ndef load(path=CONFIG):\n    with open(path) as f:\n        return json.load(f)\n\n\ndef port(cfg):\n    return cfg["server"]["port"]\n\n\nif __name__ == "__main__":\n    cfg = load()\n    print(f"listening on {port(cfg)}")\n',
    "config.json": '{\n  "server": {"host": "0.0.0.0", "port": 80},\n  "log": {"level": "debug", "file": "/var/log/app.log"}\n}\n',
    "README.md": "# app\n\nA tiny service. Run `python app.py`. Config in config.json; tests in tests/.\n",
}
TASKS = [
    "The service in /srv/app fails to start with 'permission denied' on port 80. Find out why and fix the config so it uses 8080. Verify with a test run.",
    "In /srv/app, add a function `host(cfg)` next to `port(cfg)` in app.py, use it in the print line, and run the app to check.",
    "Look through /srv/app, write a short summary of what it does to /srv/app/SUMMARY.md, then add two todo items for missing tests.",
    "Search the web for the current Python release, save the version to memory, and tell me in the chat channel 'ops'.",
]


def result(name, args):
    if name == "read_file":
        f = FILES.get(str(args.get("path", "")).split("/")[-1])
        return "\n".join(f"{i + 1}|{l}" for i, l in enumerate(f.split("\n"))) if f else "error: no such file"
    if name == "list_dir": return "app.py\nconfig.json\nREADME.md\ntests/"
    if name == "search_files": return 'app.py:11:    return cfg["server"]["port"]\nconfig.json:2:  "server": {"host": "0.0.0.0", "port": 80},'
    if name == "terminal":
        c = str(args.get("command", ""))
        if "python" in c: return "listening on 8080" if random.random() < 0.5 else "PermissionError: [Errno 13] Permission denied: ('0.0.0.0', 80)"
        if "cat" in c:
            f = next((v for k, v in FILES.items() if k in c), None)
            return f or "cat: no such file"
        return "ok"
    if name in ("write_file", "patch"): return '{"ok": true, "bytes": 412}'
    if name == "web_search": return '[{"title": "Python 3.14.1 released", "url": "https://www.python.org/downloads/"}]'
    if name == "web_extract": return "Python 3.14.1 is the latest stable release."
    if name == "todo_read": return '{"items": []}'
    return '{"ok": true}'


def post(base, body):
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


MARKUP = re.compile(r"<\|tool_call>|<tool_call\|>|<\|\"\|>|\bcall:[a-z_]+\{")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--label", required=True)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", default="/root/gbench/agent.jsonl")
    a = ap.parse_args()
    random.seed(1)
    tot = {"steps": 0, "calls": 0, "markup_as_text": 0, "text_with_calls": 0, "final": 0, "errors": 0}
    examples = []
    for run in range(a.runs):
        for ti, task in enumerate(TASKS):
            msgs = [{"role": "system", "content": "You are an autonomous coding and ops agent. Use the tools to act; say briefly what you do before each step. When finished, answer with a short summary."},
                    {"role": "user", "content": task}]
            for step in range(a.steps):
                try:
                    r = post(a.base, {"model": "x", "messages": msgs, "tools": TOOLS, "max_tokens": 2048,
                                      "temperature": a.temperature, "seed": run * 100 + ti * 10 + step})
                except Exception as e:
                    tot["errors"] += 1; examples.append(f"run{run} task{ti} step{step}: HTTP error {e}"); break
                m = r["choices"][0]["message"]; fr = r["choices"][0]["finish_reason"]
                content = m.get("content") or ""; calls = m.get("tool_calls") or []
                tot["steps"] += 1; tot["calls"] += len(calls)
                if MARKUP.search(content):
                    tot["markup_as_text"] += 1; examples.append(f"run{run} task{ti} step{step}: {content[-300:]!r}")
                if calls and content.strip(): tot["text_with_calls"] += 1
                msgs.append({"role": "assistant", "content": m.get("content"), **({"tool_calls": calls} if calls else {})})
                if not calls:
                    tot["final"] += fr == "stop"
                    break
                for c in calls:
                    try: args = json.loads(c["function"]["arguments"] or "{}")
                    except ValueError: args = {}
                    msgs.append({"role": "tool", "tool_call_id": c["id"], "content": result(c["function"]["name"], args)})
            print(f"run{run} task{ti}: {len(msgs)} messages", flush=True)
    print("AGENT", a.label, json.dumps(tot), flush=True)
    for e in examples[:6]: print("  ", e)
    with open(a.out, "a") as f:
        f.write(json.dumps({"label": a.label, "time": time.strftime("%F %T"), **tot, "examples": examples}) + "\n")


if __name__ == "__main__":
    main()
