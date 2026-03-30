#!/usr/bin/env python3
"""Temporary script to write FYP report draft. Delete after use."""
import os
import re

OUTPUT = r"C:\Users\bedor\OneDrive\Documents\FYP\Writing Journal\H00404752_Shoaeb_Draft.md"

# The report is stored in report_body.txt next to this script
BODY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_body.txt")

with open(BODY, "r", encoding="utf-8") as f:
    content = f.read()

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write(content)

wc = len(re.findall(r"\b\w+\b", content))
print(f"Written {os.path.getsize(OUTPUT):,} bytes (~{wc} words) to {OUTPUT}")
