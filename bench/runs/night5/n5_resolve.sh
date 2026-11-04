#!/bin/bash
cd /opt/NInferno-up
G() { git -c user.name="Leon Bollerup" -c user.email=mail@r2x2.com "$@"; }
python3 - <<'PY'
p = "src/models/gemma4/program_family.cpp"
s = open(p).read()
a = s.index("<<<<<<< HEAD\n")
b = s.index(">>>>>>> 7fe1dece", a)
b = s.index("\n", b) + 1
s = s[:a] + "#include <cmath>\n#include <cstdlib>\n#include <cstring>\n" + s[b:]
open(p, "w").write(s)
PY
sed -n 24,36p src/models/gemma4/program_family.cpp
grep -c "<<<<<<<\|>>>>>>>" src/models/gemma4/program_family.cpp
git add src/models/gemma4/program_family.cpp
G -c core.editor=true cherry-pick --continue > /dev/null && echo "ok #318"
G cherry-pick -x -m 1 9d8a8f29 > /dev/null && echo "ok #319"
git status --porcelain --untracked-files=no | head
git log --oneline kvshare-up3..HEAD | cut -c1-100
