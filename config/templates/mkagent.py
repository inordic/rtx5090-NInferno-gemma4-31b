"""Make the agent template from Google's 2026-07-09 Gemma 4 template: an assistant message's text is
rendered before its tool calls (the order the model generates them), and a turn whose tool results
end the conversation stays open whether or not the message had text."""
import sys

src, dst = sys.argv[1], sys.argv[2]
lines = open(src).read().split("\n")
i_set = next(i for i, l in enumerate(lines) if "{%- set captured_content -%}" in l)
i_end = next(i for i, l in enumerate(lines) if "{%- endset -%}" in l and i > i_set)
i_out = next(i for i, l in enumerate(lines) if "{{- captured_content -}}" in l)
i_calls = next(i for i, l in enumerate(lines) if "{%- if message.get('tool_calls') -%}" in l)
assert i_calls < i_set < i_end < i_out, (i_calls, i_set, i_end, i_out)
block = lines[i_set:i_end + 1]
pre = block + ["            {%- if message.get('tool_calls') -%}{{- captured_content -}}{%- endif -%}", ""]
lines[i_out] = "            {%- if not message.get('tool_calls') -%}{{- captured_content -}}{%- endif -%}"
del lines[i_set:i_end + 1]
lines[i_calls:i_calls] = pre
text = "\n".join(lines)
old_close = "{%- elif not (ns_tr_out.flag and not has_content and not next_nt.found) -%}"
assert text.count(old_close) == 1
text = text.replace(old_close, "{%- elif not (ns_tr_out.flag and not next_nt.found) -%}")
old_head = "  Context: Fixed tool-calling loops, turn closures, and thinking content-ordering.\n#}"
assert text.count(old_head) == 1
text = text.replace(old_head, old_head[:-2] + (
    "  Agent patch (rtx5090-NInferno-gemma4-31b): an assistant message's text goes before its tool calls,\n"
    "  as the model generates it, and a turn whose tool results end the conversation stays open.\n#}"))
open(dst, "w").write(text)
