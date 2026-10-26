import struct, sys
f = open(sys.argv[1], "rb")
def rd(fmt): return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))
magic, ver = f.read(4), rd("I")[0]
nt, nkv = rd("QQ")
def rstr(): n = rd("Q")[0]; return f.read(n).decode("utf8", "replace")
sizes = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
def rval(t):
    if t == 8: return rstr()
    if t == 9:
        at, n = rd("I")[0], rd("Q")[0]
        if at == 8:
            for _ in range(n): rstr()
            return f"<array str x{n}>"
        f.read(struct.calcsize(sizes[at]) * n); return f"<array t{at} x{n}>"
    return rd(sizes[t])[0]
for _ in range(nkv):
    k = rstr(); t = rd("I")[0]; v = rval(t)
    if any(s in k for s in ("vocab", "tokens", "embedding_length", "block_count")): print(k, v)
for _ in range(nt):
    name = rstr(); nd = rd("I")[0]; dims = rd("Q" * nd); tt = rd("I")[0]; off = rd("Q")[0]
    if "token_embd" in name or name.startswith("output"): print(name, dims, tt)
