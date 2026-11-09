"""Tokenising for keyword search (BM25) and the offline fake provider: lowercase words, common
English stop words dropped, and a light suffix strip so "policies"/"policy" and "booked"/"book" meet."""

from __future__ import annotations

import re

WORD = re.compile(r"[a-z0-9]+")
STOP = frozenset(
    "a an and are as at be by can do does for from how i in is it of on or our the to we what when where which who "  # noqa: SIM905
    "why with you your my me us this that there their they was were will would should could much many any".split()
)


def terms(text: str) -> list[str]:
    out = []
    for word in WORD.findall(text.lower()):
        if word in STOP:
            continue
        for suffix in ("ing", "ed", "es", "s"):
            if len(word) > 4 and word.endswith(suffix):
                word = word[: -len(suffix)]
                break
        out.append(word)
    return out
