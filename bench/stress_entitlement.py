"""stress_entitlement: conversations that branch at a turn anchor and are resumed while unrelated
sessions fill a small host cache, the shape of the 2026-10-27 engine exit (a consumed endpoint
whose long anchor a sibling branch shared, and pressure evicted the sibling).

usage: stress_entitlement.py --rounds N [--seed S] [--base URL]
Stops at the first failed request and prints FAILED; prints PASSED after all rounds."""
import argparse
import random
import sys
import time

from agentsynth import Client, system_text, text, tools

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=30)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--base", default="http://127.0.0.1:8080")
ap.add_argument("--mode", choices=["branch", "agent"], default="branch",
                help="branch: siblings at a turn anchor; agent: long tool-result loops into a full cache")
a = ap.parse_args()

c = Client(a.base)
rng = random.Random(a.seed)
SYS = system_text(11, 2500)
TOOLS = tools(11, 8, desc_words=120)
asks = "Reply with one short sentence and do not call any tool."
convs = []          # live conversations: lists of messages ending with an assistant reply
n_req = 0
t_start = time.time()


def send(messages, tag):
    global n_req
    n_req += 1
    try:
        r = c.chat(messages, TOOLS, max_tokens=12)
    except Exception as e:  # connection refused, HTTP 500: the engine is gone or refused
        print(f"FAILED request {n_req} ({tag}) after {time.time() - t_start:.0f}s: {e}", flush=True)
        sys.exit(1)
    u = r["usage"]
    print(f"{n_req:4d} {tag:9s} prompt {u.get('prompt_tokens', 0):6d} "
          f"cached {(u.get('prompt_tokens_details') or {}).get('cached_tokens', 0):6d} {r['seconds']:5.1f}s", flush=True)
    return r["content"]


def new_conv(words):
    m = [{"role": "system", "content": SYS}, {"role": "user", "content": text(rng, words) + " " + asks}]
    m.append({"role": "assistant", "content": send(m, "new")})
    return m


def extend(m, words, tag):
    m = m + [{"role": "user", "content": text(rng, words) + " " + asks}]
    m.append({"role": "assistant", "content": send(m, tag)})
    return m


if a.mode == "agent":
    # The cache full of unrelated sessions first, then a few long tool loops, mostly staying on
    # one: every step must call a tool, the call goes back exactly as returned (so the next
    # request continues the cached endpoint, as an agent's does) with a tool result of
    # 150-1800 words.
    def loop_step(m, tag):
        global n_req
        n_req += 1
        try:
            r = c.chat(m, TOOLS, max_tokens=160, tool_choice="required")
        except Exception as e:
            print(f"FAILED request {n_req} ({tag}) after {time.time() - t_start:.0f}s: {e}", flush=True)
            sys.exit(1)
        u = r["usage"]
        print(f"{n_req:4d} {tag:9s} prompt {u.get('prompt_tokens', 0):6d} "
              f"cached {(u.get('prompt_tokens_details') or {}).get('cached_tokens', 0):6d} "
              f"calls {len(r['tool_calls'])} {r['seconds']:5.1f}s", flush=True)
        m = m + [{"role": "assistant", "content": r["content"] or None,
                  **({"tool_calls": r["tool_calls"]} if r["tool_calls"] else {})}]
        if r["tool_calls"]:
            for call in r["tool_calls"]:
                m.append({"role": "tool", "tool_call_id": call["id"], "content": text(rng, rng.randint(150, 1800))})
        else:
            m.append({"role": "user", "content": text(rng, rng.randint(150, 1800)) + " Call a tool next."})
        return m

    def new_loop():
        return [{"role": "system", "content": SYS},
                {"role": "user", "content": text(rng, rng.randint(200, 800)) + " Work through this with the tools, one call per step."}]

    for _ in range(6):
        new_conv(rng.randint(1500, 4000))
    loops = [new_loop() for _ in range(3)]
    cur = 0
    for rnd in range(a.rounds):
        if rng.random() < 0.25:
            cur = rng.randrange(len(loops))
        loops[cur] = loop_step(loops[cur], f"loop{cur}")
        if rng.random() < 0.08:
            new_conv(rng.randint(1500, 4000))
        if len(loops[cur]) > 70:
            loops[cur] = new_loop()
        if (rnd + 1) % 10 == 0:
            print(f"--- step {rnd + 1}/{a.rounds}, {n_req} requests, {time.time() - t_start:.0f}s", flush=True)
    print(f"PASSED {a.rounds} steps, {n_req} requests in {time.time() - t_start:.0f}s")
    sys.exit(0)

for rnd in range(a.rounds):
    # A trunk of a few turns, so it carries turn anchors.
    if not convs or rng.random() < 0.4:
        trunk = new_conv(rng.randint(800, 2500))
        for _ in range(rng.randint(2, 4)):
            trunk = extend(trunk, rng.randint(800, 3000), "turn")
        convs.append(trunk)
    trunk = rng.choice(convs)
    # A sibling: the same history with the last user message replaced (reuses the turn anchor).
    sib = trunk[:-2] + [{"role": "user", "content": text(rng, rng.randint(200, 1500)) + " " + asks}]
    sib.append({"role": "assistant", "content": send(sib, "branch")})
    convs.append(sib)
    # Pressure: unrelated sessions into the small host cache.
    for _ in range(rng.randint(1, 4)):
        convs.append(new_conv(rng.randint(1500, 4000)))
    # Resume the trunk from its endpoint (consumes it) and sometimes the sibling too.
    idx = convs.index(trunk)
    convs[idx] = extend(trunk, rng.randint(300, 1500), "resume")
    if rng.random() < 0.5:
        j = convs.index(sib)
        convs[j] = extend(sib, rng.randint(300, 1500), "resume-sb")
    if len(convs) > 24:
        convs = convs[-24:]
    print(f"--- round {rnd + 1}/{a.rounds} done, {n_req} requests, {time.time() - t_start:.0f}s", flush=True)
print(f"PASSED {a.rounds} rounds, {n_req} requests in {time.time() - t_start:.0f}s")
