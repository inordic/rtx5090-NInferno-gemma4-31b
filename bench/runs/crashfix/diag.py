"""diag.py <tree>: checked_resource_difference names its call site and both operands when it throws."""
import sys

root = sys.argv[1]
h = f"{root}/src/models/qwen3_5/program/context_work.h"
c = f"{root}/src/models/qwen3_5/program/context_work.cpp"


def edit(path, old, new):
    s = open(path).read()
    assert s.count(old) == 1, (path, old[:50])
    open(path, "w").write(s.replace(old, new))


edit(h,
     "detail::PhysicalResources checked_resource_difference(detail::PhysicalResources value,\n"
     "                                                      detail::PhysicalResources removed);",
     "detail::PhysicalResources checked_resource_difference(\n"
     "    detail::PhysicalResources value, detail::PhysicalResources removed,\n"
     "    std::source_location where = std::source_location::current());")
s = open(h).read()
first = s.index("#include <")
open(h, "w").write(s[:first] + "#include <source_location>\n" + s[first:])

edit(c,
     "detail::PhysicalResources checked_resource_difference(detail::PhysicalResources value,\n"
     "                                                      detail::PhysicalResources removed) {",
     "detail::PhysicalResources checked_resource_difference(detail::PhysicalResources value,\n"
     "                                                      detail::PhysicalResources removed,\n"
     "                                                      std::source_location where) {")
edit(c,
     '        throw std::logic_error("Qwen3.5 resource subtraction underflow");',
     '        const auto text = [](const detail::PhysicalResources& r) {\n'
     '            return "device lanes " + std::to_string(r.device.active_lanes) + " slots " +\n'
     '                   std::to_string(r.device.state_slots) + " main " +\n'
     '                   std::to_string(r.device.main_kv_pages) + " backend " +\n'
     '                   std::to_string(r.device.backend_kv_pages) + " host slots " +\n'
     '                   std::to_string(r.host.state_slots) + " bytes " +\n'
     '                   std::to_string(r.host.kv_bytes);\n'
     '        };\n'
     '        throw std::logic_error("Qwen3.5 resource subtraction underflow at " +\n'
     '                               std::string(where.file_name()) + ":" +\n'
     '                               std::to_string(where.line()) + " | value " + text(value) +\n'
     '                               " | removed " + text(removed));')
s = open(c).read()
if "#include <string>\n" not in s:
    s = s.replace("#include <stdexcept>\n", "#include <stdexcept>\n#include <string>\n", 1)
open(c, "w").write(s)
print("diag applied")
