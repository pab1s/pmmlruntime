#!/usr/bin/env python3
"""Check the mdBook tree: SUMMARY coverage, links, and prose rules.

Run from the repo root:  python3 docs/check-docs.py

Rules enforced (see docs/src/style-guide.md):
  1. every SUMMARY.md entry exists on disk
  2. every .md under docs/src is either in SUMMARY.md or marked internal
  3. every relative .md link resolves
  4. published pages carry no em/en dashes, no spaced dash substitutes in prose,
     no product names other than pmmlruntime/JPMML/real converters
  5. published pages keep the scaffold: Info callout + Next Steps
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
INTERNAL = {"style-guide.md", "SUMMARY.md"}

DASHES = {"\u2014": "em dash", "\u2013": "en dash"}
BANNED_WORDS = [
    "additionally",
    "comprehensive",
    "crucial",
    "delve",
    "enhance",
    "ensure",
    "foster",
    "highlight",
    "intuitive",
    "leverage",
    "robust",
    "seamless",
    "showcase",
    "underscore",
    "valuable",
    "vibrant",
    "powerful",
    "effortless",
    "simply",
    "easily",
    "actually",
    "basically",
]

failures = []


def fail(msg):
    failures.append(msg)


def read(path):
    return open(path, encoding="utf-8").read()


def summary_entries():
    text = read(os.path.join(SRC, "SUMMARY.md"))
    return [
        os.path.normpath(os.path.join(SRC, m))
        for m in re.findall(r"\]\((\./[^)]+\.md)\)", text)
    ]


def check_summary(listed):
    for path in listed:
        if not os.path.isfile(path):
            fail("SUMMARY entry missing on disk: %s" % os.path.relpath(path, SRC))


def check_coverage(listed):
    for dirpath, _, files in os.walk(SRC):
        for name in files:
            if not name.endswith(".md"):
                continue
            path = os.path.normpath(os.path.join(dirpath, name))
            rel = os.path.relpath(path, SRC)
            if path not in listed and rel not in INTERNAL:
                fail("page not listed in SUMMARY.md: %s" % rel)


def check_links(listed):
    link = re.compile(r"\]\((\.[^)#]+\.md)(?:#[^)]*)?\)")
    for path in listed:
        text = read(path)
        for match in link.finditer(text):
            target = os.path.normpath(
                os.path.join(os.path.dirname(path), match.group(1))
            )
            if not os.path.isfile(target):
                fail(
                    "broken link in %s: %s"
                    % (os.path.relpath(path, SRC), match.group(1))
                )


def check_prose(listed):
    for path in listed:
        rel = os.path.relpath(path, SRC)
        text = read(path)
        in_code = False
        for number, line in enumerate(text.splitlines(), 1):
            if line.strip().startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                continue
            where = "%s:%d" % (rel, number)
            for char, label in DASHES.items():
                if char in line:
                    fail("%s %s" % (where, label))
            low = line.lower()
            if "mlflow" in low:
                fail("%s names another product" % where)
            is_table = line.lstrip().startswith("|")
            is_heading = line.lstrip().startswith("#")
            if not is_table and not is_heading and " - " in line:
                fail("%s spaced dash in prose" % where)
            for word in BANNED_WORDS:
                if re.search(r"\b%s\b" % word, low):
                    fail("%s banned word %r" % (where, word))
        if "## Next Steps" not in text:
            fail("%s has no ## Next Steps section" % rel)
        if not re.search(r"^> \*\*(Info|Note|Tip|Warning|Attention)", text, re.M):
            fail("%s has no admonition" % rel)


def main():
    listed = summary_entries()
    check_summary(listed)
    check_coverage(listed)
    check_links(listed)
    check_prose(listed)
    print("checked %d published pages" % len(listed))
    if failures:
        print("FAIL (%d)" % len(failures))
        for item in failures:
            print("  -", item)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
