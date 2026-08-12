"""Map OCR display text to canonical shift codes after recognition."""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path
from typing import Iterable, Mapping


class CodeCanonicalizer:
    def __init__(self, canonical_codes: Iterable[str], aliases: Mapping[str, str] | None = None) -> None:
        self.exact = {unicodedata.normalize("NFC", code): code for code in canonical_codes}
        self.folded = {key.casefold(): value for key, value in self.exact.items()}
        for display, canonical in (aliases or {}).items():
            normalized = unicodedata.normalize("NFC", display)
            self.exact[normalized] = canonical
            self.folded[normalized.casefold()] = canonical

    @classmethod
    def from_dictionary(cls, path: Path) -> "CodeCanonicalizer":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            codes = [row["canonical_code"] for row in csv.DictReader(stream)]
        return cls(codes)

    def convert(self, display_text: str) -> str | None:
        value = unicodedata.normalize("NFC", display_text.strip())
        return self.exact.get(value, self.folded.get(value.casefold()))

    def result(self, display_text: str) -> dict[str, str | None]:
        return {
            "display_text": unicodedata.normalize("NFC", display_text),
            "canonical_code": self.convert(display_text),
        }
