"""Render an agent-style conversation with two Gemma 4 chat templates and show where they differ."""
import json, sys
import jinja2

TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
         {"type": "function", "function": {"name": "terminal", "description": "Run a shell command",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
MSGS = [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "Check the log and fix the config."},
    {"role": "assistant", "content": "I'll read the log first.",
     "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": {"path": "/var/log/x.log"}}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "ERROR: port 80 in use"},
    {"role": "assistant", "content": "",
     "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "terminal", "arguments": {"command": "ss -ltnp | grep :80"}}}]},
    {"role": "tool", "tool_call_id": "c2", "content": "nginx 1234"},
    {"role": "assistant", "content": "Port 80 is used by nginx. Changing the config now.",
     "tool_calls": [{"id": "c3", "type": "function", "function": {"name": "terminal", "arguments": {"command": "sed -i s/80/8081/ app.conf"}}}]},
    {"role": "tool", "tool_call_id": "c3", "content": ""},
]


def render(path, msgs):
    env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
    def raise_exception(m): raise RuntimeError(m)
    env.globals["raise_exception"] = raise_exception
    t = env.from_string(open(path).read())
    return t.render(messages=msgs, tools=TOOLS, add_generation_prompt=True, bos_token="<bos>")


if __name__ == "__main__":
  for p in sys.argv[1:]:
    print("=" * 30, p)
    print(render(p, MSGS).split("<turn|>\n", 1)[1])
