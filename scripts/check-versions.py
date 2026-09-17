#!/usr/bin/env python3
"""Fail when any manifest disagrees with VERSION_NUMBER.

Run from the repo root:  python3 scripts/check-versions.py
Optionally pass a release tag to also assert the tag matches:
  python3 scripts/check-versions.py v0.1.0
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as handle:
        return handle.read()


def manifest_version(label, path, pattern):
    match = re.search(pattern, read(path), re.M)
    return label, path, match.group(1) if match else None


def main():
    version = read("VERSION_NUMBER").strip()
    found = [
        manifest_version("cargo workspace", "Cargo.toml", r'^version = "([^"]+)"'),
        manifest_version("python", "python/pyproject.toml", r'^version = "([^"]+)"'),
        manifest_version(
            "python native", "python/_native/Cargo.toml", r'^version = "([^"]+)"'
        ),
        manifest_version(
            "java",
            "java/pom.xml",
            r"<artifactId>pmmlruntime</artifactId>\s*<version>([^<]+)</version>",
        ),
        manifest_version(
            "java native", "java/native/Cargo.toml", r'^version = "([^"]+)"'
        ),
        manifest_version(
            "node", "javascript/node/package.json", r'^\s*"version":\s*"([^"]+)"'
        ),
        manifest_version(
            "web", "javascript/web/package.json", r'^\s*"version":\s*"([^"]+)"'
        ),
        manifest_version(
            "web crate", "javascript/web/Cargo.toml", r'^version = "([^"]+)"'
        ),
        manifest_version(
            "node crate", "javascript/node/Cargo.toml", r'^version = "([^"]+)"'
        ),
    ]

    failures = []
    for label, path, value in found:
        if value is None:
            failures.append(f"cannot read a version from {path} ({label})")
        elif value != version:
            failures.append(
                f"{path} ({label}) has {value}, VERSION_NUMBER has {version}"
            )

    tag = sys.argv[1] if len(sys.argv) > 1 else None
    if tag and tag != "v" + version:
        failures.append("tag %s does not match VERSION_NUMBER %s" % (tag, version))

    print("VERSION_NUMBER %s, %d manifests checked" % (version, len(found)))
    if failures:
        for item in failures:
            print("  MISMATCH:", item)
        return 1
    print("all versions agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
