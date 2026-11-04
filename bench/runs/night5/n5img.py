#!/usr/bin/env python3
"""n5img: a Chat tool result that ends in an image, what an agent sends after a screenshot tool
(upstream #293 answered 500 on Gemma 4). Prints the status and the answer."""
import json, sys, urllib.error, urllib.request
sys.path.insert(0, "/root/gbench")
from visiontest import data_uri

tools = [{"type": "function", "function": {"name": "screenshot", "description": "Take a screenshot.",
                                           "parameters": {"type": "object", "properties": {}}}}]
messages = [
    {"role": "user", "content": "Take a screenshot and tell me the number in its title."},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "screenshot", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": [
        {"type": "text", "text": "Screenshot taken."},
        {"type": "image_url", "image_url": {"url": data_uri("visual_chart.png")}}]},
]
body = {"model": "x", "messages": messages, "tools": tools, "max_tokens": 48, "temperature": 0}
req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    print("status 200:", repr((out["choices"][0]["message"].get("content") or "")[:80]))
except urllib.error.HTTPError as e:
    print("status", e.code, e.read()[:200])
