p = "/opt/NInferno/src/convert/checks.cpp"
s = open(p).read()
old = """    const auto vocab = model.config()["vocab_size"].get<std::uint64_t>();
    const auto size  = std::filesystem::file_size(ranking);
    if (size == 0 || size % (vocab * 8)) {
        throw std::runtime_error("ranking must contain complete int64 vocabulary rows");"""
assert s.count(old) == 1
new = """    // A drafter imported from GGUF keeps its vocabulary in text_config.
    const auto& config = model.config();
    const auto vocab   = config.contains("vocab_size") && config.at("vocab_size").is_number()
                             ? config.at("vocab_size").get<std::uint64_t>()
                             : config.at("text_config").at("vocab_size").get<std::uint64_t>();
    const auto size    = std::filesystem::file_size(ranking);
    if (size == 0 || size % (vocab * 8)) {
        throw std::runtime_error("ranking must contain complete int64 vocabulary rows (vocab " +
                                 std::to_string(vocab) + ", " + std::to_string(size) + " bytes)");"""
s = s.replace(old, new)
open(p, "w").write(s)
print("ok")
