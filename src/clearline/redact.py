"""Masking personal information before it reaches a model prompt or the audit log.

Covers Social Insurance Numbers (checked with the Luhn digit so ordinary 9-digit numbers are left
alone), email addresses and North American phone numbers.
"""

from __future__ import annotations

import re

SIN = re.compile(r"\b(\d{3})[ -]?(\d{3})[ -]?(\d{3})\b")
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\d)")


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def redact(text: str) -> str:
    text = EMAIL.sub("[EMAIL]", text)
    text = SIN.sub(lambda m: "[SIN]" if luhn_ok("".join(m.groups())) else m.group(0), text)
    return PHONE.sub("[PHONE]", text)
