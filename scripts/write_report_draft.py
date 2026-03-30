"""Script to write the FYP final report draft to the Writing Journal directory."""
import os
import re

OUTPUT_PATH = r"C:\Users\bedor\OneDrive\Documents\FYP\Writing Journal\H00404752_Shoaeb_Draft.md"

# Read the content from the template file next to this script
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "report_body.dat")

if os.path.exists(TEMPLATE_PATH):
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        CONTENT = f.read()
else:
    print(f"Template not found at {TEMPLATE_PATH}")
    exit(1)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(CONTENT)

# Word count analysis
lines = CONTENT.split("\n")
in_excluded = False
body_lines = []
for line in lines:
    stripped = line.strip()
    if stripped.startswith("## Abstract"):
        in_excluded = True
    elif stripped.startswith("## 1."):
        in_excluded = False
    elif stripped.startswith("## References"):
        in_excluded = True
    elif stripped.startswith("## Appendix"):
        in_excluded = True

    if not in_excluded:
        body_lines.append(line)

body_text = " ".join(body_lines)
word_count = len(re.findall(r"\b\w+\b", body_text))
total_count = len(re.findall(r"\b\w+\b", CONTENT))

print(f"File written: {OUTPUT_PATH}")
print(f"File size: {os.path.getsize(OUTPUT_PATH):,} bytes")
print(f"Body word count (excl abstract/refs/appendix): ~{word_count}")
print(f"Total word count (all content): ~{total_count}")
