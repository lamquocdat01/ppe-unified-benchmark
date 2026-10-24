"""C5 render check: count bibliography items without DOI/URL on the rendered PDF (pdftotext), not in the .bib.

Usage: python c5_render_check.py <rendered.pdf>
"""
import re
import subprocess
import sys

pdf = sys.argv[1]
txt = subprocess.run(["pdftotext", "-enc", "UTF-8", "-layout", pdf, "-"], capture_output=True).stdout.decode("utf-8", "replace")
parts = re.split(r"\n\s*\[(\d+)\]\s", "\n" + txt)
items = [(parts[k], re.sub(r"\s+", "", parts[k + 1])) for k in range(1, len(parts) - 1, 2)]
bad = [n for n, body in items if not re.search(r"doi:|https?://|10\.\d{4,}/", body)]
print(f"rendered items: {len(items)}; without DOI/URL: {len(bad)} {bad}")
sys.exit(1 if bad else 0)
