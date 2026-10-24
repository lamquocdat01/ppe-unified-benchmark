"""List the \\TBD cells of manuscript v5: every use of a numbers.tex macro whose value is wrapped in \\TBD.

Usage:
  python -u scripts/tbd_list.py <latex_source dir> <out.md>
"""
import re
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
nt = (src / "numbers.tex").read_text(encoding="utf-8")
defined = set(re.findall(r"\\newcommand\{\\(n[A-Za-z]+)\}", nt))
tbd = dict(re.findall(r"\\newcommand\{\\(n[A-Za-z]+)\}\{\\TBD\{([^}]*)\}\}", nt))
tex = (src / "main.tex").read_text(encoding="utf-8").split("\n")

uses, used = [], set()
section = ""
for i, line in enumerate(tex, 1):
    m = re.match(r"\\(sub)*section\*?\{(.+?)\}", line.strip())
    if m:
        section = m.group(2)
    for name in re.findall(r"\\(n[A-Za-z]+)", line):
        if name in defined:
            used.add(name)
        if name in tbd:
            uses.append((i, section, name, tbd[name]))

undefined = sorted({n for line in tex for n in re.findall(r"\\(n[A-Z][A-Za-z]+|n[a-z]+[A-Z][A-Za-z]*)", line)}
                   - defined - {"newcommand", "newsavebox", "newenvironment"})
rows = ["# TBD_LIST v5 (sinh bằng `scripts/tbd_list.py` từ main.tex + numbers.tex)", "",
        f"- Macro số định nghĩa: {len(defined)}; dùng trong main.tex: {len(used)}.",
        f"- Ô `\\TBD` trong main.tex: **{len(uses)} lần dùng**, {len({u[2] for u in uses})} macro khác nhau.",
        f"- Macro TBD định nghĩa nhưng chưa dùng: {', '.join(sorted(set(tbd) - used)) or 'không'}.",
        "", "| Dòng main.tex | Mục | Macro | Giá trị hiện in |", "|---|---|---|---|"]
rows += [f"| {i} | {s} | `\\{n}` | {v} |" for i, s, n, v in uses]
out.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"{len(uses)} TBD uses, {len({u[2] for u in uses})} macros; undefined-looking names: {undefined[:10]}")
