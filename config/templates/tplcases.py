"""Render conversation shapes with Google's template and the agent patch; print which differ."""
import sys
from tplrender import render

call = lambda i, n, a: {"id": i, "type": "function", "function": {"name": n, "arguments": a}}
U = {"role": "user", "content": "Hi, what is 2+2?"}
S = {"role": "system", "content": "You are helpful."}
CASES = {
    "plain chat": [S, U],
    "two turns": [S, U, {"role": "assistant", "content": "4."}, {"role": "user", "content": "And 3+3?"}],
    "call only, result last": [S, U, {"role": "assistant", "content": None, "tool_calls": [call("a", "terminal", {"command": "ls"})]},
                               {"role": "tool", "tool_call_id": "a", "content": "x"}],
    "call, result, final answer, user": [S, U, {"role": "assistant", "content": "", "tool_calls": [call("a", "terminal", {"command": "ls"})]},
                                         {"role": "tool", "tool_call_id": "a", "content": "x"},
                                         {"role": "assistant", "content": "Done."}, {"role": "user", "content": "thanks"}],
    "text+call, result last": [S, U, {"role": "assistant", "content": "Checking.", "tool_calls": [call("a", "terminal", {"command": "ls"})]},
                               {"role": "tool", "tool_call_id": "a", "content": "x"}],
    "text+call, result, user": [S, U, {"role": "assistant", "content": "Checking.", "tool_calls": [call("a", "terminal", {"command": "ls"})]},
                                {"role": "tool", "tool_call_id": "a", "content": "x"}, {"role": "user", "content": "go on"}],
    "parallel calls": [S, U, {"role": "assistant", "content": "", "tool_calls": [call("a", "terminal", {"command": "ls"}), call("b", "read_file", {"path": "/a"})]},
                       {"role": "tool", "tool_call_id": "a", "content": "x"}, {"role": "tool", "tool_call_id": "b", "content": "y"}],
}
for name, msgs in CASES.items():
    a, b = render(sys.argv[1], msgs), render(sys.argv[2], msgs)
    print(f"{'same' if a == b else 'DIFFERENT'}: {name}")
    if a != b:
        print("   google:", repr(a.split("<turn|>\n", 2)[-1][-260:]))
        print("   agent: ", repr(b.split("<turn|>\n", 2)[-1][-260:]))
