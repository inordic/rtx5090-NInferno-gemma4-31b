import sys
root = "/opt/NInferno/src/ops/linear/q6/"
variant = sys.argv[1]
# Launchers for SIMT T7 / T8 (kept for both variants).
p = root + "q6_instances.cuh"; s = open(p).read()
if "SimtR8T8" not in s:
    s = s.replace("using SimtR8T7 = Q6A16SimtSchedule<8, 7, 1, 16, 2, Cache::ca, 1>;\n",
                  "using SimtR8T7 = Q6A16SimtSchedule<8, 7, 1, 16, 2, Cache::ca, 1>;\n"
                  "using SimtR8T8 = Q6A16SimtSchedule<8, 8, 1, 16, 2, Cache::ca, 1>;\n")
    open(p, "w").write(s)
p = root + "q6_a16_simt.cu"; s = open(p).read()
if "launch_q6_a16_simt_r8_t7" not in s:
    add = ""
    for t in (7, 8):
        add += f"""
void launch_q6_a16_simt_r8_t{t}(const Tensor& x, const Weight& weight, Tensor& out,
                              cudaStream_t stream) {{
    launch_q6_a16_simt<q6_instances::SimtR8T{t}>(
        q6_linear_operands(x, weight),
        LinearBf16Output{{static_cast<__nv_bfloat16*>(out.data), weight.n}}, LinearIdentityEpilogue{{}},
        stream);
}}
"""
    s = s.replace("\n} // namespace ninferno::ops::detail", add + "\n} // namespace ninferno::ops::detail")
    open(p, "w").write(s)
p = root + "q6_launch.h"; s = open(p).read()
if "launch_q6_a16_simt_r8_t7" not in s:
    anchor = "void launch_q6_a16_simt_r8_t6(const Tensor& x, const Weight& weight, Tensor& out,"
    i = s.index(anchor); j = s.index(";", i) + 1
    decl = s[i:j]
    s = s[:j] + "\n" + decl.replace("t6", "t7") + "\n" + decl.replace("t6", "t8") + s[j:]
    open(p, "w").write(s)
# The head's selection for T 7..8.
p = root + "shapes/n262144_k5376.cpp"; s = open(p).read()
old = "    if (tokens <= 6) return launch_q6_a16_simt_r8_t6;\n"
assert old in s
if variant == "A":
    new = old + "    if (tokens <= 7) return launch_q6_a16_simt_r8_t7;\n    if (tokens <= 8) return launch_q6_a16_simt_r8_t8;\n"
else:
    new = old + "    if (tokens <= 8 && build_features().mma_tensor) return launch_q6_a16_sliced_r16_t8_w4_s2;\n"
s = s.replace(old, new)
open(p, "w").write(s)
print("patched", variant)
