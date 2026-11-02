"""fix.py <tree>: a capture group that would publish a shared copy taking more KV pages out of the
request's active entitlement than it holds publishes privately instead (capture.cpp)."""
import sys

p = f"{sys.argv[1]}/src/models/qwen3_5/program/transactions/capture.cpp"
s = open(p).read()


def edit(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)


edit("    const bool publish_shared = group.shared && permit_shared_publication &&\n"
     "                                exact_shared == nullptr && shared_prefix_capacity != 0;",
     "    bool publish_shared = group.shared && permit_shared_publication && exact_shared == nullptr &&\n"
     "                          shared_prefix_capacity != 0;")
edit("            active_removed.device.backend_kv_pages = backend_snapshot_shape->unique_full_pages;\n"
     "            added.device.backend_kv_pages          = backend_snapshot_shape->copied_pages();\n"
     "        }\n"
     "    }\n",
     "            active_removed.device.backend_kv_pages = backend_snapshot_shape->unique_full_pages;\n"
     "            added.device.backend_kv_pages          = backend_snapshot_shape->copied_pages();\n"
     "        }\n"
     "        // A conversation that continues from its own checkpoint keeps the checkpoint's prefix\n"
     "        // outside its active entitlement (request_plan: the retained prefix). Since the shared\n"
     "        // candidate after the last message joins the turn closure's group at the closed turn's\n"
     "        // end (DECISIONS 63), pages of that prefix the active sequence alone still references\n"
     "        // (its checkpoint's KV on Host, or released) count among its unique pages; a shared copy\n"
     "        // would take more out of the entitlement than it holds. The group publishes privately.\n"
     "        const detail::PhysicalResources& entitled = requests[lane].active_resources;\n"
     "        if (active_removed.device.main_kv_pages > entitled.device.main_kv_pages ||\n"
     "            active_removed.device.backend_kv_pages > entitled.device.backend_kv_pages) {\n"
     "            static std::atomic<std::uint64_t> reported{0};\n"
     "            const std::uint64_t count = ++reported;\n"
     "            if (count <= 20 || count % 1000 == 0) {\n"
     "                std::fprintf(stderr,\n"
     "                             \"ninferno: the capture at %u publishes privately (%llu): a shared \"\n"
     "                             \"copy takes %u main KV pages, the request is entitled to %u\\n\",\n"
     "                             static_cast<unsigned>(group.frontier),\n"
     "                             static_cast<unsigned long long>(count),\n"
     "                             active_removed.device.main_kv_pages, entitled.device.main_kv_pages);\n"
     "            }\n"
     "            publish_shared              = false;\n"
     "            assessment.publishes_shared = false;\n"
     "            active_removed              = {};\n"
     "            added                       = {};\n"
     "            text_snapshot_shape.reset();\n"
     "            backend_snapshot_shape.reset();\n"
     "            if (!publish_private) {\n"
     "                assessment.physically_feasible = true;\n"
     "                return assessment;\n"
     "            }\n"
     "        }\n"
     "    }\n")
for inc in ("#include <atomic>\n", "#include <cstdio>\n"):
    if inc not in s:
        first = s.index("#include <")
        s = s[:first] + inc + s[first:]
open(p, "w").write(s)
print("fix applied")
