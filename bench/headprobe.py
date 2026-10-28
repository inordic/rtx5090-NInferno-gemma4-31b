"""headprobe: two new sessions with the same long synthetic system head and 20 tools; the second
should reuse the head from the cache."""
import random

from agentsynth import Client, system_text, text, tools

c = Client()
sys_a, t20 = system_text(101, 13000), tools(201, 20, 260)
rng = random.Random(99)
for i in range(3):
    m = [{"role": "system", "content": sys_a},
         {"role": "user", "content": f"Probe run {i}. " + text(rng, 120) + " Reply with one short sentence."}]
    r = c.chat(m, t20, max_tokens=16)
    u = r["usage"]
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    print(f"new session {i}: prompt {u.get('prompt_tokens')} cached {cached} {r['seconds']:.2f}s")
