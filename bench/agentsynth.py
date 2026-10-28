"""agentsynth: deterministic synthetic agent prompts (system texts, tool sets, turns) and a small
OpenAI chat client, shared by stress_entitlement.py and cachereplay.py. No real traffic is used."""
import json
import random
import time
import urllib.error
import urllib.request

WORDS = ("the of and to in is that for it as with was on be by this are or at from have an they which "
         "one you were all we when there can been has more if will would each about how up out them then "
         "she many some so these would other into has her two like him see time could no make than first "
         "been its who now people my made over did down only way find use may water long little very after "
         "words called just where most know get through back much before go good new write our used me man "
         "too any day same right look think also around another came come work three word must because does "
         "part even place well such here take why help put different away again off went old number great "
         "tell men say small every found still between name should home big give air line set own under read "
         "last never us left end along while might next sound below saw something thought both few those "
         "always looked show large often together asked house world going want school important until form "
         "food keep children feet land side without boy once animals life enough took sometimes four head "
         "above kind began almost live page got earth need far hand high year mother light parts country "
         "father let night following picture being study second eyes soon times story boys since white days "
         "ever paper hard near sentence better best across during today others however sure means knew its "
         "try told young miles sun ways thing whole hear example heard several change answer room sea against "
         "top turned learn point city play toward five using himself usually money seen car morning given "
         "server request config deploy cache token memory schedule report status check update file folder "
         "agent task result error retry backup monitor alert metric queue worker build release branch").split()


def text(rng, n_words):
    out, cap = [], True
    for i in range(n_words):
        w = rng.choice(WORDS)
        out.append(w.capitalize() if cap else w)
        cap = False
        if rng.random() < 0.07:
            out[-1] += "."
            cap = True
        elif rng.random() < 0.05:
            out[-1] += ","
        if rng.random() < 0.02:
            out[-1] += "\n"
    return " ".join(out)


def system_text(seed, n_words, shared_with=None, shared_frac=0.0):
    """A system prompt; with shared_with, its first shared_frac of words equal that seed's text."""
    if shared_with is None:
        return text(random.Random(seed), n_words)
    head = text(random.Random(shared_with), int(n_words * shared_frac))
    return head + " " + text(random.Random(seed), n_words - int(n_words * shared_frac))


def tools(seed, n, desc_words=260):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        props = {f"arg_{j}": {"type": "string", "description": text(rng, 18)} for j in range(4)}
        out.append({"type": "function", "function": {
            "name": f"tool_{seed}_{i}", "description": text(rng, desc_words),
            "parameters": {"type": "object", "properties": props, "required": ["arg_0"]}}})
    return out


class Client:
    def __init__(self, base="http://127.0.0.1:8080", model="m"):
        self.base, self.model = base, model

    def chat(self, messages, tool_list=None, max_tokens=16, timeout=600, tool_choice="none"):
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": 0, "stream": False}
        if tool_list:
            body["tools"], body["tool_choice"] = tool_list, tool_choice
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
        msg = out["choices"][0]["message"]
        return {"content": msg.get("content") or "", "tool_calls": msg.get("tool_calls") or [],
                "usage": out.get("usage", {}), "seconds": time.time() - t0}
