#!/usr/bin/env python
import subprocess
import sys

commands = [
    "git status",
    "git diff",
    "git log --oneline -10",
]

for cmd in commands:
    print(f"\n{'=' * 60}")
    print(f"Running: {cmd}")
    print("=" * 60)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
