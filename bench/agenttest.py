#!/usr/bin/env python3
"""agenttest: an agent loop against the server - 20 tools, simulated results, the assistant's messages
sent back as an OpenAI client does (text and calls together). Counts steps where tool markup came
back as text instead of tool_calls, and prints a summary row.
The simulated workspace keeps state per task: write_file and patch change what read_file, cat and a test run see,
and the test run's outcome follows the files (config port, app.py compiling; nothing the model writes is run).
Arguments are checked against each tool's schema: malformed JSON or a schema violation is counted and answered
with an error, never treated as {}. Unknown tools are counted and answered with an error. Each task's outcome is
checked at the end. Exit code 0 only with no transport errors, no markup as text, no unknown tools, no invalid
arguments and at least --min-tasks tasks done; 1 otherwise."""
import argparse, copy, json, re, sys, time, urllib.request

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
SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
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


def type_ok(v, spec):
    t = spec.get("type")
    if t == "string": return isinstance(v, str)
    if t == "integer": return isinstance(v, int) and not isinstance(v, bool)
    if t == "number": return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "boolean": return isinstance(v, bool)
    if t == "object": return isinstance(v, dict)
    if t == "array":
        if not isinstance(v, list): return False
        item = spec.get("items")
        if not item: return True
        for x in v:
            if not type_ok(x, item): return False
            if item.get("type") == "object":
                for k, sub in item.get("properties", {}).items():
                    if k in x and not type_ok(x[k], sub): return False
        return True
    return True


def validate(name, text):
    """(args, problem, extra): problem is None when the arguments are a JSON object that fits the tool's schema;
    extra lists arguments the schema does not name (allowed, since the schemas don't forbid them, but counted)."""
    try:
        args = json.loads(text or "{}")
    except ValueError:
        return None, "arguments are not valid JSON", []
    if not isinstance(args, dict):
        return None, "arguments are not a JSON object", []
    spec = SCHEMAS[name]
    for k in spec.get("required", []):
        if k not in args:
            return args, f"missing required argument {k!r}", []
    props = spec.get("properties", {})
    for k, v in args.items():
        if k in props and not type_ok(v, props[k]):
            return args, f"argument {k!r} should be of type {props[k].get('type')}", []
    return args, None, [k for k in args if k not in props]


class Workspace:
    """One task's simulated /srv/app, memory, todo list and chat."""

    def __init__(self):
        self.files, self.memory, self.todos, self.sent, self.runs = copy.deepcopy(FILES), {}, [], [], []

    def run_app(self):
        try:
            cfg = json.loads(self.files.get("config.json", ""))
            port, host = cfg["server"]["port"], cfg["server"].get("host", "0.0.0.0")
        except (ValueError, KeyError, TypeError) as e:
            return f"Traceback (most recent call last):\n  ...\njson.decoder.JSONDecodeError: {e}"
        try:
            compile(self.files.get("app.py", ""), "app.py", "exec")      # compiled only, never run
        except SyntaxError as e:
            return f'  File "app.py", line {e.lineno}\nSyntaxError: {e.msg}'
        if not isinstance(port, int) or port < 1024:
            return f"PermissionError: [Errno 13] Permission denied: ('{host}', {port})"
        app = self.files["app.py"]
        uses_host = re.search(r"def host\(cfg\)", app) and re.search(r"print\(.*host\(cfg\)", app)
        return f"listening on {host}:{port}" if uses_host else f"listening on {port}"

    def result(self, name, a):
        f = lambda p: str(p or "").rstrip("/").split("/")[-1]
        if name == "read_file":
            t = self.files.get(f(a.get("path")))
            return "\n".join(f"{i + 1}|{l}" for i, l in enumerate(t.split("\n"))) if t is not None else "error: no such file"
        if name == "write_file":
            self.files[f(a["path"])] = a.get("content", "")
            return json.dumps({"ok": True, "bytes": len(a.get("content", "").encode())})
        if name == "patch":
            name_ = f(a["path"]); t = self.files.get(name_)
            if t is None: return "error: no such file"
            for e in a["edits"]:
                old, new = e.get("old", ""), e.get("new", "")
                if not old or old not in t: return f"error: text not found: {old[:60]!r}"
                t = t.replace(old, new, 1)
            self.files[name_] = t
            return json.dumps({"ok": True, "edits": len(a["edits"])})
        if name == "list_dir": return "\n".join(sorted(self.files) + ["tests/"])
        if name == "search_files":
            try: rx = re.compile(a["pattern"])
            except re.error as e: return f"error: bad regex: {e}"
            hits = [f"{n}:{i + 1}:{l}" for n, t in sorted(self.files.items()) for i, l in enumerate(t.split("\n")) if rx.search(l)]
            return "\n".join(hits[:50]) or "no matches"
        if name == "terminal":
            c = str(a.get("command", ""))
            if re.search(r"\bpython3?\b.*app\.py", c):
                out = self.run_app(); self.runs.append(out); return out
            if re.search(r"\b(cat|head|less)\b", c):
                t = next((v for k, v in self.files.items() if k in c), None)
                return t if t is not None else "cat: no such file"
            if re.search(r"\bls\b", c): return "\n".join(sorted(self.files) + ["tests/"])
            return "ok"
        if name == "python_exec": return "ok (simulated: code is not run)"
        if name == "web_search": return '[{"title": "Python 3.14.1 released", "url": "https://www.python.org/downloads/"}]'
        if name == "web_extract": return "Python 3.14.1 is the latest stable release."
        if name == "memory_save":
            self.memory[str(a["key"])] = str(a.get("value", "")); return '{"ok": true}'
        if name == "memory_search":
            q = str(a["query"]).lower()
            return json.dumps({k: v for k, v in self.memory.items() if q in k.lower() or q in v.lower()})
        if name == "todo_write":
            self.todos = list(a["items"]); return json.dumps({"ok": True, "items": len(self.todos)})
        if name == "todo_read": return json.dumps({"items": self.todos})
        if name == "send_message":
            self.sent.append(a); return '{"ok": true}'
        if name == "git_status": return "nothing to commit, working tree clean"
        if name == "git_diff": return ""
        return '{"ok": true}'

    def task_done(self, ti):
        """Whether the task's outcome is in the workspace, checked from state, not from what the model said."""
        if ti == 0:
            try: port = json.loads(self.files["config.json"])["server"]["port"]
            except (ValueError, KeyError, TypeError): return False
            return port == 8080 and any(r.startswith("listening on") for r in self.runs)
        if ti == 1:
            app = self.files.get("app.py", "")
            try: compile(app, "app.py", "exec")
            except SyntaxError: return False
            return bool(re.search(r"def host\(cfg\)", app) and re.search(r"print\(.*host\(cfg\)", app)
                        and any(r.startswith("listening on") for r in self.runs))
        if ti == 2:
            return len(self.files.get("SUMMARY.md", "").strip()) >= 40 and len(self.todos) >= 2
        if ti == 3:
            return (any("3.14" in v for v in self.memory.values())
                    and any(str(m.get("channel", "")).lower().lstrip("#") == "ops" and "3.14" in str(m.get("text", ""))
                            for m in self.sent))
        return False


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
    ap.add_argument("--min-tasks", type=int, default=0, help="tasks that must be done (of runs x 4) for exit code 0")
    ap.add_argument("--out", default="/root/gbench/agent.jsonl")
    a = ap.parse_args()
    tot = {"steps": 0, "calls": 0, "markup_as_text": 0, "text_with_calls": 0, "final": 0, "errors": 0,
           "unknown_tool": 0, "invalid_args": 0, "extra_args": 0, "tasks": 0, "tasks_done": 0}
    examples, per_task = [], []
    for run in range(a.runs):
        for ti, task in enumerate(TASKS):
            ws = Workspace()
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
                    name = c["function"]["name"]
                    if name not in SCHEMAS:
                        tot["unknown_tool"] += 1; examples.append(f"run{run} task{ti} step{step}: unknown tool {name!r}")
                        out = f"error: unknown tool {name!r}"
                    else:
                        args, problem, extra = validate(name, c["function"].get("arguments"))
                        tot["extra_args"] += len(extra)
                        if problem:
                            tot["invalid_args"] += 1
                            examples.append(f"run{run} task{ti} step{step}: {name}: {problem}: {str(c['function'].get('arguments'))[:120]!r}")
                            out = f"error: invalid arguments: {problem}"
                        else:
                            out = ws.result(name, args)
                    msgs.append({"role": "tool", "tool_call_id": c["id"], "content": out})
            done = ws.task_done(ti)
            tot["tasks"] += 1; tot["tasks_done"] += done; per_task.append(int(done))
            print(f"run{run} task{ti}: {len(msgs)} messages, task {'done' if done else 'NOT done'}", flush=True)
    ok = (tot["errors"] == 0 and tot["markup_as_text"] == 0 and tot["unknown_tool"] == 0
          and tot["invalid_args"] == 0 and tot["tasks_done"] >= a.min_tasks)
    print("AGENT", a.label, json.dumps(tot), "OK" if ok else "FAIL", flush=True)
    for e in examples[:8]: print("  ", e)
    with open(a.out, "a") as f:
        f.write(json.dumps({"label": a.label, "time": time.strftime("%F %T"), **tot, "per_task": per_task,
                            "examples": examples}) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
