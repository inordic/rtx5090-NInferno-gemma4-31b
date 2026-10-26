import json, sys
label = ""
for l in open(sys.argv[1]):
    if l.startswith("=== "):
        label = l.strip()[14:]
    if l.startswith("{"):
        r = json.loads(l)
        print("%-44s d=%s ok=%s ptok=%s pp=%s tg=%s" % (label[:44], r["depth"], r["ok"], r["ptok"], r["pp"], r["tg"]))
