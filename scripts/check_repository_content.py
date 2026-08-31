#!/usr/bin/env python3
"""CI guard: reject captured traffic and generated model archives in a diff."""
from __future__ import annotations
import os, subprocess, sys

base=os.environ.get("BASE_SHA")
head=os.environ.get("HEAD_SHA", "HEAD")
if not base:
    base=subprocess.check_output(["git","rev-parse", "HEAD^"], text=True).strip()
changed=subprocess.check_output(["git","diff","--name-only", f"{base}..{head}"],text=True).splitlines()
bad=[p for p in changed if p.endswith((".pcap",".pcapng",".keras"))]
tracked=subprocess.check_output(["git","ls-files","data/cicids2017"],text=True).splitlines()
if bad or tracked:
    if bad: print("disallowed generated/capture content in diff:", *bad, sep="\n")
    if tracked: print("disallowed tracked CICIDS2017 data:", *tracked, sep="\n")
    sys.exit(1)
