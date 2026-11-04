"""n5_l59.py <tree>: on the staged prompt path a prefill call that is neither the prompt's last nor
ends on a capture's frontier reads nothing of the last layer but its keys (patch 19).
Opt-in: NINFERNO_GEMMA4_SKIP_LAST_LAYER=1."""
import sys

root = sys.argv[1]


def edit(path, pairs):
    p = f"{root}/{path}"
    s = open(p).read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:70], s.count(old))
        s = s.replace(old, new)
    open(p, "w").write(s)


edit("src/models/qwen3_5/program/context.h", [(
    "    qwen3_5::DFlashPrefillIngress* dflash_prefill_host_ingress = nullptr;\n};\n\nstruct OrdinaryBatchContext {",
    "    qwen3_5::DFlashPrefillIngress* dflash_prefill_host_ingress = nullptr;\n"
    "    // The staged prompt path: only the prompt's last call and a call ending on a capture's\n"
    "    // frontier have their hidden read. A rebuild reads every call's tail and a causal score\n"
    "    // every column, so they leave this false.\n"
    "    bool hidden_at_frontiers_only = false;\n};\n\nstruct OrdinaryBatchContext {")])

edit("src/models/qwen3_5/program/prefill.cpp", [(
    "            0,\n            dflash_prefill_host_ingress};\n"
    "        // Forced tokens of another lane, or another lane's bind, may have repointed the prefill KV\n",
    "            0,\n            dflash_prefill_host_ingress,\n            true};\n"
    "        // Forced tokens of another lane, or another lane's bind, may have repointed the prefill KV\n")])

edit("src/models/gemma4/target.h", [
    ("                   const FeatureTap* tap = nullptr, const Tensor* embeddings = nullptr) const;\n\n"
     "    // Soft-capped head over the columns of `normed`",
     "                   const FeatureTap* tap = nullptr, const Tensor* embeddings = nullptr,\n"
     "                   bool output = true) const;\n\n"
     "    // Soft-capped head over the columns of `normed`"),
    ("                const FeatureTap* tap, const Tensor* ple) const;\n",
     "                const FeatureTap* tap, const Tensor* ple, bool output) const;\n"),
    ("    // (BF16 [hidden, T]) replaces the ids' embedding rows: an image's soft tokens.\n",
     "    // (BF16 [hidden, T]) replaces the ids' embedding rows: an image's soft tokens. With `output`\n"
     "    // false the caller reads nothing of the result: the last layer stops at its cache write.\n"),
])

edit("src/models/gemma4/target.cu", [
    ("                        const FeatureTap* tap, const Tensor* ple) const {\n    const cudaStream_t s     = device_.stream;",
     "                        const FeatureTap* tap, const Tensor* ple, bool output) const {\n    const cudaStream_t s     = device_.stream;"),
    ("                                        qkv_released);\n            }\n            Tensor context = work_.alloc(DType::BF16, {d, c_.query_heads, t});\n",
     "                                        qkv_released);\n            }\n"
     "            // Its keys are in the cache; the rest of the last layer only feeds the result.\n"
     "            if (!output && i + 1 == c_.layers) { return; }\n"
     "            Tensor context = work_.alloc(DType::BF16, {d, c_.query_heads, t});\n"),
    ("                           const FeatureTap* tap, const Tensor* embeddings) const {\n    const std::int32_t t = ids.ne[0];",
     "                           const FeatureTap* tap, const Tensor* embeddings, bool output) const {\n    const std::int32_t t = ids.ne[0];"),
    ("    layers(work, x, h, pos, kv, envelope, activations, tap, c_.ple() ? &ple : nullptr);\n    return h;",
     "    layers(work, x, h, pos, kv, envelope, activations, tap, c_.ple() ? &ple : nullptr, output);\n    return h;"),
])

edit("src/models/gemma4/program_family.cpp", [(
    "    Tensor hidden = core.prefill_hidden.slice(1, 0, t);\n"
    "    {\n"
    "        auto scope          = core.work.scope();\n"
    "        const Tensor normed = pass_.forward(\n"
    "            core.work, device_ids, positions,\n"
    "            {.layers = layers, .image_block = embeddings != nullptr},\n"
    "            {.max_visible_keys = base + len, .max_tokens = t}, Activations::Recorded, nullptr,\n"
    "            embeddings);\n"
    "        CUDA_CHECK(cudaMemcpyAsync(hidden.data, normed.data, hidden.bytes(),\n"
    "                                   cudaMemcpyDeviceToDevice, s));\n"
    "    }\n",
    "    // Mid-prompt, nothing reads this call's hidden: the last layer is a full-attention layer\n"
    "    // that no later layer attends through, so only its keys matter. A call ending on a capture's\n"
    "    // frontier keeps the output: finish_prefill_text stores its tail in the checkpoint.\n"
    "    static const bool skip_last_layer = [] {\n"
    "        const char* e = std::getenv(\"NINFERNO_GEMMA4_SKIP_LAST_LAYER\");\n"
    "        return e != nullptr && std::atoi(e) != 0;\n"
    "    }();\n"
    "    const bool stores_tail = split_frontier && *split_frontier == base + len &&\n"
    "                             state.rewrite_checkpoint_hidden != nullptr;\n"
    "    const bool output = !skip_last_layer || !state.hidden_at_frontiers_only || is_last || stores_tail;\n"
    "    Tensor hidden = core.prefill_hidden.slice(1, 0, t);\n"
    "    {\n"
    "        auto scope          = core.work.scope();\n"
    "        const Tensor normed = pass_.forward(\n"
    "            core.work, device_ids, positions,\n"
    "            {.layers = layers, .image_block = embeddings != nullptr},\n"
    "            {.max_visible_keys = base + len, .max_tokens = t}, Activations::Recorded, nullptr,\n"
    "            embeddings, output);\n"
    "        if (output) {\n"
    "            CUDA_CHECK(cudaMemcpyAsync(hidden.data, normed.data, hidden.bytes(),\n"
    "                                       cudaMemcpyDeviceToDevice, s));\n"
    "        }\n"
    "    }\n")])
print("patch 19 applied")
