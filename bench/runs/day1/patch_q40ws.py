p = "/opt/NInferno/src/ops/generic/generic_gemm.cpp"
s = open(p).read()
old = """bool mmq_prefill_wins(const Problem& p, const Weight& w) {
    return mmq_prefill_device() && ggml_mmq_format(p.qtype) &&"""
assert s.count(old) == 1
s = s.replace(old, """// Experiment: NINFERNO_Q40_W4A8_WS=1 sends GGML Q4_0 prefill on sm_120 to the producer-warp W4A8
// tile (w4a8_ws_wins) instead of ggml's MMQ.
bool q40_w4a8_ws() {
    static const bool on = [] {
        const char* e = std::getenv("NINFERNO_Q40_W4A8_WS");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

bool mmq_prefill_wins(const Problem& p, const Weight& w) {
    if (p.qtype == QType::GGML_Q4_0 && q40_w4a8_ws()) return false;
    return mmq_prefill_device() && ggml_mmq_format(p.qtype) &&""")
old2 = """    case QType::GGML_Q4_0:
        return ada && p.n >= 5376 && p.k >= 5376;"""
assert s.count(old2) == 1
s = s.replace(old2, """    case QType::GGML_Q4_0:
        return (ada && p.n >= 5376 && p.k >= 5376) || (blackwell && q40_w4a8_ws());""")
if "#include <cstdlib>" not in s:
    assert "#include <algorithm>" in s
    s = s.replace("#include <algorithm>", "#include <algorithm>\n#include <cstdlib>", 1)
open(p, "w").write(s)
print("ok")
