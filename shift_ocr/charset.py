"""Versioned fixed Korean CTC charset and dataset coverage checks."""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping


def load_charset(path: Path) -> list[str]:
    """Load literal characters and immutable ``@range HEX-HEX`` directives."""
    characters: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip("\ufeff")
        if not line or line.startswith("#"):
            continue
        if line.startswith("@range "):
            first, last = line[7:].split("-", 1)
            characters.extend(chr(codepoint) for codepoint in range(int(first, 16), int(last, 16) + 1))
        elif line.startswith("@chars "):
            characters.extend(list(line[7:]))
        else:
            characters.append(line)
    if len(characters) != len(set(characters)):
        duplicates = [char for char, count in Counter(characters).items() if count > 1]
        raise ValueError(f"charset contains duplicate entries: {duplicates[:10]}")
    return characters


def normalize_transcription(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def coverage_report(records: Iterable[Mapping[str, object]], charset: Iterable[str]) -> dict[str, object]:
    supported = set(charset)
    frequencies: Counter[str] = Counter()
    oov: Counter[str] = Counter()
    non_nfc = 0
    record_count = 0
    code_oov: Counter[str] = Counter()
    for record in records:
        text = str(record.get("display_text", record.get("display_code", "")))
        normalized = normalize_transcription(text)
        record_count += 1
        non_nfc += text != normalized
        frequencies.update(normalized)
        missing = [char for char in normalized if char not in supported]
        oov.update(missing)
        if record.get("canonical_code") is not None:
            code_oov.update(missing)
    total_characters = sum(frequencies.values())
    return {
        "record_count": record_count,
        "character_count": total_characters,
        "non_nfc_records": non_nfc,
        "oov_count": sum(oov.values()),
        "oov_rate": sum(oov.values()) / max(1, total_characters),
        "code_oov_count": sum(code_oov.values()),
        "oov_characters": dict(sorted(oov.items())),
        "character_frequency": dict(sorted(frequencies.items())),
    }


def write_coverage_report(report: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if int(report["code_oov_count"]) != 0:
        raise ValueError("shift-code OOV must be zero")
