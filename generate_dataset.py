#!/usr/bin/env python3
"""Synthetic nurse shift-schedule dataset generator.

Outputs
-------
- One Excel workbook containing schedule sheets, name/code dictionaries,
  row-level ground truth, cell-level ground truth, manifest, and sources.
- One clean PNG per schedule.
- JSONL/CSV annotations for row-level and cell-level OCR training.

Important name-data limitation
------------------------------
Public online annual Korean baby-name rankings are readily available from
2008 onward. This generator targets worker birth years 1966-2007, so it uses
published historical anchor-year rankings (1968/1978/1988/1998/2008) and maps
nearby birth years to the closest anchor. The output records provenance fields
so these fallback names are never presented as exact annual rankings.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import math
import random
import re
import statistics
import sys
import textwrap
from functools import lru_cache
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import requests
    from bs4 import BeautifulSoup
except Exception:  # scraper is optional; built-in fallback still works
    requests = None
    BeautifulSoup = None

try:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
except Exception as exc:
    raise RuntimeError('Pillow is required. Install with: pip install Pillow') from exc

try:
    from artifact_tool import Workbook, SpreadsheetFile
except Exception as exc:
    raise RuntimeError(
        'artifact_tool is required for Excel export in the Codex/ChatGPT environment.'
    ) from exc

CURRENT_YEAR = 2026
MIN_BIRTH_YEAR = 1966  # age 60 in 2026
MAX_BIRTH_YEAR = 2007  # age 19 in 2026

SOURCE_BABY_NAME = 'https://baby-name.kr/annalRanking/{year}/{page}'
SOURCE_HISTORY = 'https://www.yna.co.kr/view/AKR20250707129900518'
SOURCE_SURNAME_KOSIS = 'https://kosis.kr/statisticsList/mass/mass_list.jsp?org_id=101&process=statHtml&tbl_id=DT_1IN15SD'
SOURCE_SURNAME_MIRROR = 'https://www.rootsinfo.co.kr/info/roots/table_sung15.php'
SOURCE_TEMPLATE_PDF = 'user-uploaded: 26년 3월 근무 편성표.pdf'

# Top-5 factual historical anchor rankings published from court/family-register data.
HISTORICAL_NAME_ANCHORS: dict[int, dict[str, list[str]]] = {
    1968: {
        'male': ['성호', '영수', '영호', '영철', '정호'],
        'female': ['미경', '미숙', '경희', '경숙', '영숙'],
    },
    1978: {
        'male': ['정훈', '성훈', '상훈', '성진', '지훈'],
        'female': ['지영', '은정', '미영', '현정', '은주'],
    },
    1988: {
        'male': ['지훈', '성민', '현우', '정훈', '동현'],
        'female': ['지혜', '지은', '수진', '혜진', '은지'],
    },
    1998: {
        'male': ['동현', '지훈', '성민', '현우', '준호'],
        'female': ['유진', '민지', '수빈', '지원', '지현'],
    },
    2008: {
        'male': ['민준', '지훈', '현우', '준서', '우진'],
        'female': ['서연', '민서', '지민', '서현', '서윤'],
    },
}

# Curated supplements expand the synthetic pool while retaining period-like style.
# They are explicitly marked as curated_supplement in the generated dictionary.
ERA_SUPPLEMENTS: dict[int, dict[str, list[str]]] = {
    1968: {
        'male': ['영진', '정수', '성수', '병철', '동수', '재호', '태수', '광수', '상철', '종수'],
        'female': ['정희', '영희', '명숙', '정숙', '순희', '옥희', '경자', '영자', '순자', '복순'],
    },
    1978: {
        'male': ['상민', '재훈', '정호', '영훈', '경수', '진호', '태훈', '동훈', '성수', '준호'],
        'female': ['미정', '선영', '혜정', '영미', '수정', '경미', '은영', '정은', '희정', '현주'],
    },
    1988: {
        'male': ['민수', '성현', '태현', '진우', '정민', '승현', '재현', '상현', '민석', '동욱'],
        'female': ['지현', '현정', '민정', '소영', '유정', '은정', '수연', '지영', '혜진', '나영'],
    },
    1998: {
        'male': ['민재', '승현', '동욱', '재민', '정우', '태현', '준영', '건우', '성준', '우진'],
        'female': ['수민', '예진', '채원', '서영', '다은', '은지', '소현', '유진', '주연', '가영'],
    },
    2008: {
        'male': ['예준', '현준', '도현', '준혁', '민재', '승현', '민성', '승민', '건우', '동현'],
        'female': ['예은', '하은', '수빈', '지우', '유진', '은서', '민지', '윤서', '예진', '지원',
                   '예원', '수민', '가은', '수연', '채원', '다은', '수아', '유빈', '유나', '채린'],
    },
}

# Offline fallback: top 100 entries from the 2015 Korean census surname ranking.
# Online runs attempt to scrape every published row (532 rows) from SOURCE_SURNAME_MIRROR.
# Same Hangul surname with different Hanja is kept as a separate source row, then aggregated
# by Hangul spelling for random sampling.
SURNAME_FALLBACK_2015: list[tuple[str, str, int]] = [
    ('김', '金', 10689959), ('이', '李', 7306828), ('박', '朴', 4192074),
    ('최', '崔', 2333927), ('정', '鄭', 2151879), ('강', '姜', 1176847),
    ('조', '趙', 1055567), ('윤', '尹', 1020547), ('장', '張', 992721),
    ('임', '林', 823921), ('한', '韓', 773404), ('오', '吳', 763281),
    ('서', '徐', 751704), ('신', '申', 741081), ('권', '權', 705941),
    ('황', '黃', 697171), ('안', '安', 685639), ('송', '宋', 683494),
    ('전', '全', 559110), ('홍', '洪', 558853), ('유', '柳', 478990),
    ('고', '高', 471396), ('문', '文', 464040), ('양', '梁', 460600),
    ('손', '孫', 457303), ('배', '裵', 400641), ('조', '曺', 398260),
    ('백', '白', 381986), ('허', '許', 326770), ('유', '劉', 302511),
    ('남', '南', 275648), ('심', '沈', 271749), ('노', '盧', 256229),
    ('정', '丁', 243803), ('하', '河', 230481), ('곽', '郭', 203188),
    ('성', '成', 199124), ('차', '車', 194782), ('주', '朱', 194766),
    ('우', '禹', 194713), ('구', '具', 193080), ('신', '辛', 192877),
    ('임', '任', 191261), ('전', '田', 186469), ('민', '閔', 171740),
    ('유', '兪', 167927), ('류', '柳', 163703), ('나', '羅', 160946),
    ('진', '陳', 157599), ('지', '池', 153491), ('엄', '嚴', 144425),
    ('채', '蔡', 131557), ('원', '元', 129522), ('천', '千', 121780),
    ('방', '方', 94831), ('공', '孔', 91869), ('강', '康', 91625),
    ('현', '玄', 88824), ('함', '咸', 80659), ('변', '卞', 78156),
    ('염', '廉', 69387), ('양', '楊', 69101), ('변', '邊', 60633),
    ('여', '呂', 60522), ('추', '秋', 60483), ('노', '魯', 58698),
    ('도', '都', 56850), ('소', '蘇', 52427), ('신', '愼', 51865),
    ('석', '石', 49203), ('선', '宣', 42733), ('설', '薛', 42646),
    ('마', '馬', 38949), ('길', '吉', 38173), ('주', '周', 37240),
    ('연', '延', 34766), ('방', '房', 33520), ('위', '魏', 31342),
    ('표', '表', 30743), ('명', '明', 29110), ('기', '奇', 28829),
    ('반', '潘', 28062), ('라', '羅', 25960), ('왕', '王', 25565),
    ('금', '琴', 25432), ('옥', '玉', 25107), ('육', '陸', 23455),
    ('인', '印', 22363), ('맹', '孟', 22028), ('제', '諸', 21976),
    ('모', '牟', 21534), ('장', '蔣', 21508), ('남궁', '南宮', 21308),
    ('탁', '卓', 21099), ('국', '鞠', 20547), ('여', '余', 20134),
    ('진', '秦', 19301), ('어', '魚', 18849), ('은', '殷', 16894),
    ('편', '片', 16689),
]

SHIFT_CODE_GROUPS: dict[str, list[str]] = {
    'core': [
        'D', 'E', 'N', 'M', 'MD', 'MID', 'S', 'A', 'P', 'AM', 'PM',
        'D5', 'LD', 'DE', 'D/E', 'DB', 'NB', 'DN', 'D12', 'N12',
    ],
    'off': [
        'O', 'OFF', 'Off', 'off', 'OF', 'X', 'NO', 'DO', 'RD', 'SD', 'AO', 'RO',
        'WO', 'PO', 'ADO', 'H', 'PH', 'NA', '-', '—',
    ],
    'leave': [
        'AL', 'A/L', 'ANL', 'VAC', 'V', 'VL', 'LV', 'HL', 'H+', 'H⁺', 'AMH', 'PMH',
        'OH', 'SH', 'EH', 'BH', 'UN', 'XA', 'DH', 'IH', 'SL', 'S/L', 'ML', 'MAT',
        'PL', 'DL', 'BL', 'CL', 'CCL', 'UPL', 'NPL', 'UL', 'EL', 'STL', 'MC',
        'BDO', 'TOIL', 'C',
    ],
    'training': [
        'T', 'TR', 'TRN', 'EDU', 'ST', 'EXT', 'INT', 'MTG', 'CONF', 'SEM', 'WS',
    ],
    'oncall': ['OC', 'ON', 'CALL', 'O/C', 'ONC', 'SB', 'STBY'],
    'role': [
        'IC', 'CH', 'CN', 'TL', 'OT', 'OR', 'OPD', 'ER', 'ED', 'ICU', 'PACU',
        'CSSD', 'CB', 'HELP', 'SUP', 'HLP', 'F', 'f',
    ],
    'korean': [
        '주', '중', '생', '생휴', '연', '연차', '연가', '휴', '휴가', '휴무', '비번',
        '반', '반차', '공', '공가', '교', '교육', '경', '경조', '경조사', '보',
        '보건', '보건휴가', '병', '병가', '출', '출산', '출산휴가', '육', '육아',
        '육아휴직', '특', '특휴', '특별휴가', '노', '노조', '노조휴가', '휴직',
        '출장', '회의', '당', '당직', '대기', '지원', '파견',
    ],
}

GROUP_WEIGHTS = {
    'core': 0.52,
    'off': 0.25,
    'leave': 0.08,
    'training': 0.04,
    'oncall': 0.025,
    'role': 0.035,
    'korean': 0.05,
}

TEMPLATE_IDS = [
    'compact_summary',
    'clean_grid',
    'highlighted_grid',
    'grouped_hospital',
    'parted_pdf',
]

WEEKDAY_KO = ['월', '화', '수', '목', '금', '토', '일']


def dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


ALL_SHIFT_CODES = dedupe(code for group in SHIFT_CODE_GROUPS.values() for code in group)


@dataclass
class NameEntry:
    birth_year: int
    gender: str
    rank: int
    given_name: str
    source_year: int
    source_method: str
    source_url: str


@dataclass
class SurnameEntry:
    rank: int
    surname: str
    hanja: str
    population: int
    source_year: int = 2015
    source_method: str = 'scraped_2015_census_mirror'
    source_url: str = SOURCE_SURNAME_MIRROR


@dataclass
class SurnameSample:
    surname: str
    population: int
    best_rank: int
    hanja_variants: str
    source_method: str
    source_url: str


@dataclass
class PersonRow:
    row_id: str
    name: str
    given_name: str
    surname: str
    surname_rank: int
    surname_population: int
    surname_hanja_variants: str
    surname_source_method: str
    surname_source_url: str
    birth_year: int
    gender: str
    group: str
    codes_canonical: list[str]
    codes_display: list[str]
    excel_row: int = 0
    name_cell: str = ''


@dataclass
class ScheduleRecord:
    schedule_id: str
    template_id: str
    year: int
    month: int
    day_count: int
    weekdays: list[str]
    rows: list[PersonRow]
    title: str
    sheet_name: str
    xlsx_path: str = ''
    clean_image_path: str = ''
    image_width: int = 0
    image_height: int = 0
    cell_annotations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GeneratorConfig:
    count: int = 5
    seed: int = 20260723
    output_dir: str = 'output'
    min_people: int = 18
    max_people: int = 32
    min_birth_year: int = MIN_BIRTH_YEAR
    max_birth_year: int = MAX_BIRTH_YEAR
    female_ratio: float = 0.88
    case_mutation_probability: float = 0.45
    ensure_all_codes: bool = True
    scrape_recent_names: bool = False
    recent_name_top_n: int = 100
    scrape_surnames: bool = True
    surname_max_rank: int = 532
    surname_min_population: int = 5
    surname_weight_power: float = 0.75
    template_ids: list[str] = field(default_factory=lambda: TEMPLATE_IDS.copy())
    fixed_months: list[int] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: str | Path) -> 'GeneratorConfig':
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(**data)


def nearest_anchor(year: int) -> int:
    return min(HISTORICAL_NAME_ANCHORS, key=lambda anchor: abs(anchor - year))


def parse_name_ranking_html(html: str, year: int, limit: int = 100) -> list[NameEntry]:
    """Best-effort parser for baby-name.kr annual ranking pages.

    The site's structure may change. This parser accepts table rows containing
    rank/name/count and identifies gender from the nearest heading.
    """
    if BeautifulSoup is None:
        return []
    soup = BeautifulSoup(html, 'html.parser')
    entries: list[NameEntry] = []
    gender = 'unknown'
    for element in soup.find_all(['h2', 'h3', 'table', 'tr']):
        text = ' '.join(element.get_text(' ', strip=True).split())
        if '남자 이름' in text:
            gender = 'male'
        elif '여자 이름' in text:
            gender = 'female'
        if element.name != 'tr' or gender == 'unknown':
            continue
        cells = [' '.join(x.get_text(' ', strip=True).split()) for x in element.find_all(['th', 'td'])]
        if len(cells) < 2:
            continue
        rank_match = re.search(r'\d+', cells[0].replace(',', ''))
        name_match = re.fullmatch(r'[가-힣]{1,5}', cells[1])
        if not rank_match or not name_match:
            continue
        rank = int(rank_match.group())
        if rank > limit:
            continue
        entries.append(NameEntry(
            birth_year=year,
            gender=gender,
            rank=rank,
            given_name=cells[1],
            source_year=year,
            source_method='annual_web_ranking',
            source_url=SOURCE_BABY_NAME.format(year=year, page=1),
        ))
    unique: dict[tuple[str, int], NameEntry] = {}
    for entry in entries:
        unique[(entry.gender, entry.rank)] = entry
    return sorted(unique.values(), key=lambda x: (x.gender, x.rank))


def scrape_annual_names(year: int, limit: int = 100, timeout: int = 12) -> list[NameEntry]:
    if requests is None or BeautifulSoup is None or year < 2008:
        return []
    found: list[NameEntry] = []
    max_pages = max(1, math.ceil(limit / 100)) + 1
    for page in range(1, max_pages + 1):
        url = SOURCE_BABY_NAME.format(year=year, page=page)
        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers={'User-Agent': 'Mozilla/5.0 synthetic-dataset-generator/1.0'},
            )
            response.raise_for_status()
        except Exception:
            break
        page_entries = parse_name_ranking_html(response.text, year, limit)
        if not page_entries:
            break
        found.extend(page_entries)
        if len(found) >= limit * 2:
            break
    unique: dict[tuple[str, str], NameEntry] = {}
    for entry in found:
        unique[(entry.gender, entry.given_name)] = entry
    return sorted(unique.values(), key=lambda x: (x.gender, x.rank))


def build_name_dictionary(
    min_year: int = MIN_BIRTH_YEAR,
    max_year: int = MAX_BIRTH_YEAR,
    scrape_recent: bool = False,
    recent_top_n: int = 100,
) -> list[NameEntry]:
    """Build birth-year name pools with explicit provenance.

    1966-2007 are mapped to the nearest historical anchor year because a
    complete public annual ranking source for those years was not identified.
    """
    entries: list[NameEntry] = []
    recent_cache: dict[int, list[NameEntry]] = {}
    if scrape_recent:
        # Used as an optional source for 2007 fallback and future extensions.
        recent_cache[2008] = scrape_annual_names(2008, recent_top_n)

    for year in range(min_year, max_year + 1):
        anchor = nearest_anchor(year)
        source = HISTORICAL_NAME_ANCHORS[anchor]
        supplement = ERA_SUPPLEMENTS[anchor]
        for gender in ('female', 'male'):
            names = dedupe(source[gender] + supplement[gender])
            # For 2007, prefer scraped 2008 annual names when available.
            if year == 2007 and recent_cache.get(2008):
                scraped = [x.given_name for x in recent_cache[2008] if x.gender == gender]
                names = dedupe(scraped + names)
            for rank, name in enumerate(names, start=1):
                method = 'historical_anchor_nearest'
                source_url = SOURCE_HISTORY
                source_year = anchor
                if year == 2007 and recent_cache.get(2008) and name in {
                    x.given_name for x in recent_cache[2008] if x.gender == gender
                }:
                    method = 'nearest_available_annual_ranking'
                    source_url = SOURCE_BABY_NAME.format(year=2008, page=1)
                    source_year = 2008
                elif name in supplement[gender]:
                    method = 'curated_supplement'
                entries.append(NameEntry(
                    birth_year=year,
                    gender=gender,
                    rank=rank,
                    given_name=name,
                    source_year=source_year,
                    source_method=method,
                    source_url=source_url,
                ))
    return entries


def parse_surname_ranking_html(html: str) -> list[SurnameEntry]:
    """Parse ranked Korean surname rows from the 2015 census mirror page."""
    if BeautifulSoup is None:
        return []
    soup = BeautifulSoup(html, 'html.parser')
    found: dict[int, SurnameEntry] = {}

    pattern = re.compile(
        r'^\s*(\d{1,3})\s*([가-힣]{1,4})\s*\(([^)\n]+)\)\s*([\d,]+)\s*$'
    )

    # Primary path: table rows.
    for tr in soup.find_all('tr'):
        cells = [' '.join(x.get_text(' ', strip=True).split()) for x in tr.find_all(['th', 'td'])]
        joined = ' '.join(cells)
        match = pattern.match(joined)
        if not match:
            compact = ''.join(cells)
            match = pattern.match(compact)
        if not match:
            continue
        rank = int(match.group(1))
        found[rank] = SurnameEntry(
            rank=rank,
            surname=match.group(2),
            hanja=match.group(3),
            population=int(match.group(4).replace(',', '')),
        )

    # Secondary path: plain-text lines. This matches the current mirror layout.
    plain = soup.get_text('\n')
    for line in plain.splitlines():
        match = pattern.match(' '.join(line.split()))
        if not match:
            continue
        rank = int(match.group(1))
        found.setdefault(rank, SurnameEntry(
            rank=rank,
            surname=match.group(2),
            hanja=match.group(3),
            population=int(match.group(4).replace(',', '')),
        ))

    return [found[k] for k in sorted(found)]


def scrape_surname_rankings(timeout: int = 15) -> list[SurnameEntry]:
    """Scrape every available 2015 surname-ranking row.

    The official underlying statistic is KOSIS table DT_1IN15SD. The mirror page
    is used because it exposes all ranked rows in a simple HTML table.
    """
    if requests is None or BeautifulSoup is None:
        return []
    try:
        response = requests.get(
            SOURCE_SURNAME_MIRROR,
            timeout=timeout,
            headers={'User-Agent': 'Mozilla/5.0 synthetic-dataset-generator/1.1'},
        )
        response.raise_for_status()
        if not response.encoding or response.encoding.lower() in {'iso-8859-1', 'latin-1'}:
            response.encoding = response.apparent_encoding or 'euc-kr'
    except Exception:
        return []
    entries = parse_surname_ranking_html(response.text)
    if len(entries) < 50:
        return []
    return entries


def build_surname_dictionary(
    scrape: bool = True,
    max_rank: int = 532,
    min_population: int = 5,
) -> list[SurnameEntry]:
    entries = scrape_surname_rankings() if scrape else []
    if not entries:
        entries = [
            SurnameEntry(
                rank=rank,
                surname=surname,
                hanja=hanja,
                population=population,
                source_method='bundled_fallback_top100',
                source_url=SOURCE_SURNAME_MIRROR,
            )
            for rank, (surname, hanja, population) in enumerate(SURNAME_FALLBACK_2015, start=1)
        ]
    return [
        entry for entry in entries
        if entry.rank <= max_rank and entry.population >= min_population
    ]


def aggregate_surnames(entries: Sequence[SurnameEntry]) -> list[SurnameSample]:
    """Aggregate same-Hangul surnames with multiple Hanja source rows."""
    grouped: dict[str, list[SurnameEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.surname, []).append(entry)

    result: list[SurnameSample] = []
    for surname, rows in grouped.items():
        rows = sorted(rows, key=lambda x: x.rank)
        result.append(SurnameSample(
            surname=surname,
            population=sum(x.population for x in rows),
            best_rank=min(x.rank for x in rows),
            hanja_variants='|'.join(dedupe(x.hanja for x in rows)),
            source_method=rows[0].source_method,
            source_url=rows[0].source_url,
        ))
    return sorted(result, key=lambda x: (x.best_rank, -x.population, x.surname))


def weighted_surname(
    rng: random.Random,
    surname_pool: Sequence[SurnameSample],
    weight_power: float = 0.75,
) -> SurnameSample:
    if not surname_pool:
        raise ValueError('surname_pool is empty')
    weights = [max(1.0, float(item.population) ** weight_power) for item in surname_pool]
    return rng.choices(list(surname_pool), weights=weights, k=1)[0]


def mutate_ascii_case(code: str, rng: random.Random, probability: float) -> str:
    if not any('A' <= ch.upper() <= 'Z' for ch in code):
        return code
    if rng.random() >= probability:
        return code
    chars: list[str] = []
    for ch in code:
        if 'A' <= ch.upper() <= 'Z':
            chars.append(ch.upper() if rng.random() < 0.5 else ch.lower())
        else:
            chars.append(ch)
    return ''.join(chars)


def choose_code(rng: random.Random, previous: str | None = None) -> str:
    # Plausible transition bias for common D/E/N/OFF patterns.
    if previous:
        normalized = previous.upper()
        transitions = {
            'D': ['D', 'D', 'E', 'OFF', 'O', 'F'],
            'E': ['E', 'E', 'N', 'OFF', 'O', 'F'],
            'N': ['N', 'N', 'OFF', 'NO', 'O', 'F'],
            'OFF': ['D', 'E', 'N', 'OFF', 'O'],
            'O': ['D', 'E', 'N', 'OFF', 'O'],
            'F': ['D', 'E', 'N', 'F', 'f'],
        }
        if normalized in transitions and rng.random() < 0.74:
            return rng.choice(transitions[normalized])
    groups = list(GROUP_WEIGHTS)
    weights = [GROUP_WEIGHTS[g] for g in groups]
    group = rng.choices(groups, weights=weights, k=1)[0]
    return rng.choice(SHIFT_CODE_GROUPS[group])


def excel_column_name(index_1based: int) -> str:
    result = ''
    n = index_1based
    while n:
        n, rem = divmod(n - 1, 26)
        result = chr(65 + rem) + result
    return result


def schedule_day_start_col_1based(schedule: ScheduleRecord) -> int:
    return 3 if schedule.template_id in {'compact_summary', 'grouped_hospital', 'parted_pdf'} else 2


def pick_given_name(
    rng: random.Random,
    pools: dict[tuple[int, str], list[str]],
    birth_year: int,
    gender: str,
) -> str:
    candidates = pools.get((birth_year, gender)) or []
    if not candidates:
        anchor = nearest_anchor(birth_year)
        candidates = HISTORICAL_NAME_ANCHORS[anchor][gender]
    # Rank-biased choice: early names have higher probability.
    weights = [1 / (1 + i * 0.18) for i in range(len(candidates))]
    return rng.choices(candidates, weights=weights, k=1)[0]


def generate_unique_name(
    rng: random.Random,
    pools: dict[tuple[int, str], list[str]],
    surname_pool: Sequence[SurnameSample],
    surname_weight_power: float,
    birth_year: int,
    gender: str,
    used: set[str],
) -> tuple[str, SurnameSample, str]:
    for _ in range(1000):
        surname_sample = weighted_surname(rng, surname_pool, surname_weight_power)
        given = pick_given_name(rng, pools, birth_year, gender)
        full = surname_sample.surname + given
        if full not in used:
            used.add(full)
            return full, surname_sample, given

    # Deterministic suffix fallback for extremely large schedules.
    surname_sample = weighted_surname(rng, surname_pool, surname_weight_power)
    given = pick_given_name(rng, pools, birth_year, gender)
    base = surname_sample.surname + given
    i = 2
    while f'{base}{i}' in used:
        i += 1
    full = f'{base}{i}'
    used.add(full)
    return full, surname_sample, given


def schedule_month(rng: random.Random, fixed_months: Sequence[int]) -> tuple[int, int, int]:
    month = rng.choice(list(fixed_months)) if fixed_months else rng.randint(1, 12)
    days = calendar.monthrange(CURRENT_YEAR, month)[1]
    return CURRENT_YEAR, month, days


def make_group_labels(template_id: str, people_count: int) -> list[str]:
    if template_id in {'grouped_hospital', 'parted_pdf'}:
        part_size = max(5, round(people_count / 3))
        return [f'{min(i // part_size + 1, 3)}파트' for i in range(people_count)]
    if template_id == 'compact_summary':
        return [str((i % 6) + 1) for i in range(people_count)]
    return ['병동' for _ in range(people_count)]


def generate_schedule(
    schedule_index: int,
    config: GeneratorConfig,
    rng: random.Random,
    name_entries: list[NameEntry],
    surname_pool: Sequence[SurnameSample],
    forced_template: str | None = None,
) -> ScheduleRecord:
    template_id = forced_template or rng.choice(config.template_ids)
    year, month, day_count = schedule_month(rng, config.fixed_months)
    weekdays = [WEEKDAY_KO[calendar.weekday(year, month, day)] for day in range(1, day_count + 1)]
    people_count = rng.randint(config.min_people, config.max_people)
    groups = make_group_labels(template_id, people_count)

    pools: dict[tuple[int, str], list[str]] = {}
    for entry in name_entries:
        pools.setdefault((entry.birth_year, entry.gender), []).append(entry.given_name)

    used_names: set[str] = set()
    rows: list[PersonRow] = []
    for row_index in range(people_count):
        birth_year = rng.randint(config.min_birth_year, config.max_birth_year)
        gender = 'female' if rng.random() < config.female_ratio else 'male'
        full_name, surname_sample, given = generate_unique_name(
            rng,
            pools,
            surname_pool,
            config.surname_weight_power,
            birth_year,
            gender,
            used_names,
        )
        canonical: list[str] = []
        display: list[str] = []
        previous: str | None = None
        for _day in range(day_count):
            code = choose_code(rng, previous)
            canonical.append(code)
            display.append(mutate_ascii_case(code, rng, config.case_mutation_probability))
            previous = code
        rows.append(PersonRow(
            row_id=f'S{schedule_index:04d}_R{row_index + 1:03d}',
            name=full_name,
            given_name=given,
            surname=surname_sample.surname,
            surname_rank=surname_sample.best_rank,
            surname_population=surname_sample.population,
            surname_hanja_variants=surname_sample.hanja_variants,
            surname_source_method=surname_sample.source_method,
            surname_source_url=surname_sample.source_url,
            birth_year=birth_year,
            gender=gender,
            group=groups[row_index],
            codes_canonical=canonical,
            codes_display=display,
        ))

    schedule_id = f'schedule_{schedule_index:04d}'
    return ScheduleRecord(
        schedule_id=schedule_id,
        template_id=template_id,
        year=year,
        month=month,
        day_count=day_count,
        weekdays=weekdays,
        rows=rows,
        title=f'[ {year}. {month}월 ] 합성 병동 근무표',
        sheet_name=schedule_id[:31],
    )


def ensure_code_coverage(schedules: list[ScheduleRecord], rng: random.Random, probability: float) -> None:
    positions: list[tuple[ScheduleRecord, PersonRow, int]] = []
    for schedule in schedules:
        for row in schedule.rows:
            for day_index in range(schedule.day_count):
                positions.append((schedule, row, day_index))
    if len(positions) < len(ALL_SHIFT_CODES):
        raise ValueError(
            f'Not enough cells ({len(positions)}) to cover {len(ALL_SHIFT_CODES)} codes.'
        )
    rng.shuffle(positions)
    for code, (_schedule, row, day_index) in zip(ALL_SHIFT_CODES, positions):
        row.codes_canonical[day_index] = code
        row.codes_display[day_index] = mutate_ascii_case(code, rng, probability)


@lru_cache(maxsize=64)
def find_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend([
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
            '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf',
        ])
    candidates.extend([
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        'C:/Windows/Fonts/malgun.ttf',
        'C:/Windows/Fonts/malgunbd.ttf' if bold else 'C:/Windows/Fonts/malgun.ttf',
    ])
    for path in candidates:
        try:
            if Path(path).exists():
                return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str,
                  font: ImageFont.ImageFont, fill: str = '#111111') -> None:
    x1, y1, x2, y2 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text(((x1 + x2 - width) / 2, (y1 + y2 - height) / 2 - 1), text, font=font, fill=fill)


def fitted_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str,
                fill: str = '#111111', max_size: int = 12, min_size: int = 5) -> None:
    """Draw the complete label without truncating the annotation target."""
    x1, y1, x2, y2 = box
    max_w = max(1, x2 - x1 - 2)
    max_h = max(1, y2 - y1 - 2)
    for size in range(max_size, min_size - 1, -1):
        font = find_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_w and bbox[3] - bbox[1] <= max_h:
            centered_text(draw, box, text, font, fill)
            return
    # Two-line fallback for long Korean/English labels.
    split_at = max(1, len(text) // 2)
    lines = [text[:split_at], text[split_at:]]
    for size in range(min(8, max_size), min_size - 1, -1):
        font = find_font(size)
        widths = []
        heights = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            widths.append(bbox[2] - bbox[0])
            heights.append(bbox[3] - bbox[1])
        if max(widths, default=0) <= max_w and sum(heights) + 1 <= max_h:
            total_h = sum(heights) + 1
            cursor_y = y1 + (max_h - total_h) / 2
            for line, width, height in zip(lines, widths, heights):
                draw.text((x1 + (x2 - x1 - width) / 2, cursor_y), line, font=font, fill=fill)
                cursor_y += height + 1
            return
    centered_text(draw, box, text, find_font(min_size), fill)


def render_schedule_png(schedule: ScheduleRecord, output_path: Path, rng: random.Random) -> None:
    template = schedule.template_id
    people = len(schedule.rows)
    days = schedule.day_count

    if template == 'compact_summary':
        left_group, name_w, day_w, summary_w = 42, 90, 27, 44
        top_margin, title_h, header_h, row_h = 18, 30, 48, 25
        summary_cols = ['D', 'E', 'N', 'OFF', '연차']
        width = left_group + name_w + days * day_w + len(summary_cols) * summary_w + 36
        height = top_margin + title_h + header_h + people * row_h + 30
    elif template in {'grouped_hospital', 'parted_pdf'}:
        left_group, name_w, day_w = 92, 86, 28
        top_margin, title_h, note_h, header_h, row_h = 18, 34, 30, 52, 27
        summary_cols = ['D', 'E', 'N', 'OFF'] if template == 'grouped_hospital' else []
        summary_w = 42
        width = left_group + name_w + days * day_w + len(summary_cols) * summary_w + 38
        height = top_margin + title_h + note_h + header_h + people * row_h + 32
    else:
        left_group, name_w, day_w = 0, 92, 34
        top_margin, title_h, header_h, row_h = 25, 0, 50, 29
        summary_cols = []
        summary_w = 0
        width = name_w + days * day_w + 60
        height = top_margin + header_h + people * row_h + 45

    scale = 1
    image = Image.new('RGB', (width * scale, height * scale), '#F8F8F7')
    draw = ImageDraw.Draw(image)
    font = find_font(14)
    small = find_font(12)
    tiny = find_font(10)
    bold = find_font(15, bold=True)
    title_font = find_font(19, bold=True)

    annotations: list[dict[str, Any]] = []
    x0 = 18
    y = top_margin

    if template == 'compact_summary':
        table_x = x0
        table_w = left_group + name_w + days * day_w + len(summary_cols) * summary_w
        draw.rectangle((table_x, y, table_x + table_w, y + title_h), fill='#F8B800', outline='#111111', width=2)
        centered_text(draw, (table_x, y, table_x + table_w, y + title_h), schedule.title + ' - OFF : 자동', title_font)
        y += title_h
    elif template in {'grouped_hospital', 'parted_pdf'}:
        table_x = x0
        table_w = left_group + name_w + days * day_w + len(summary_cols) * summary_w
        title_fill = '#DCECD5' if template == 'parted_pdf' else '#F8B800'
        draw.rectangle((table_x, y, table_x + table_w, y + title_h), fill=title_fill, outline='#111111', width=2)
        centered_text(draw, (table_x, y, table_x + table_w, y + title_h), schedule.title, title_font)
        y += title_h
        draw.rectangle((table_x, y, table_x + table_w, y + note_h), fill='#FFFFFF', outline='#222222', width=1)
        note = '공지사항 : 합성 데이터 / 실제 직원 정보 아님 / 모든 코드는 학습용 무작위 생성'
        draw.text((table_x + 8, y + 7), note, font=tiny, fill='#333333')
        y += note_h
    else:
        table_x = x0
        table_w = name_w + days * day_w

    header_y = y
    if template in {'grouped_hospital', 'parted_pdf'}:
        draw.rectangle((table_x, header_y, table_x + left_group, header_y + header_h), fill='#DCECD5', outline='#222222')
        centered_text(draw, (table_x, header_y, table_x + left_group, header_y + header_h), '구분', bold)
        name_x = table_x + left_group
    elif template == 'compact_summary':
        draw.rectangle((table_x, header_y, table_x + left_group, header_y + header_h), fill='#F2F2F2', outline='#222222')
        centered_text(draw, (table_x, header_y, table_x + left_group, header_y + header_h), '병동', bold)
        name_x = table_x + left_group
    else:
        name_x = table_x

    draw.rectangle((name_x, header_y, name_x + name_w, header_y + header_h), fill='#F1F4F5', outline='#333333')
    centered_text(draw, (name_x, header_y, name_x + name_w, header_y + header_h), '성명', bold)
    day_start_x = name_x + name_w

    for day in range(1, days + 1):
        cell_x = day_start_x + (day - 1) * day_w
        weekday = schedule.weekdays[day - 1]
        fill = '#FFF6B0' if weekday in {'토', '일'} and template == 'highlighted_grid' else '#F6F8F8'
        draw.rectangle((cell_x, header_y, cell_x + day_w, header_y + header_h), fill=fill, outline='#8A8A8A')
        centered_text(draw, (cell_x, header_y, cell_x + day_w, header_y + header_h // 2), str(day), font)
        weekday_color = '#D11919' if weekday == '일' else ('#1357B7' if weekday == '토' else '#222222')
        centered_text(draw, (cell_x, header_y + header_h // 2, cell_x + day_w, header_y + header_h), weekday, small, weekday_color)

    summary_start_x = day_start_x + days * day_w
    for j, label in enumerate(summary_cols):
        sx = summary_start_x + j * summary_w
        draw.rectangle((sx, header_y, sx + summary_w, header_y + header_h), fill='#FAFAFA', outline='#333333')
        centered_text(draw, (sx, header_y, sx + summary_w, header_y + header_h), label, tiny)

    body_y = header_y + header_h
    group_palette = ['#F6E5B9', '#EBCFDB', '#CDE2F2', '#E7E7E7']
    group_ranges: dict[str, tuple[int, int]] = {}
    for idx, row in enumerate(schedule.rows):
        row_y = body_y + idx * row_h
        if row.group not in group_ranges:
            group_ranges[row.group] = (idx, idx)
        else:
            group_ranges[row.group] = (group_ranges[row.group][0], idx)

        if template in {'grouped_hospital', 'parted_pdf', 'compact_summary'}:
            gx1 = table_x
            gx2 = table_x + left_group
            group_index = int(re.sub(r'\D', '', row.group) or '1') - 1
            fill = group_palette[group_index % len(group_palette)]
            draw.rectangle((gx1, row_y, gx2, row_y + row_h), fill=fill, outline='#666666')

        name_fill = '#FFFFFF'
        if template == 'parted_pdf':
            group_index = int(re.sub(r'\D', '', row.group) or '1') - 1
            name_fill = group_palette[group_index % len(group_palette)]
        elif template == 'compact_summary':
            name_fill = '#FFF8E8'
        draw.rectangle((name_x, row_y, name_x + name_w, row_y + row_h), fill=name_fill, outline='#777777')
        centered_text(draw, (name_x, row_y, name_x + name_w, row_y + row_h), row.name, font)
        name_box = (name_x, row_y, name_x + name_w, row_y + row_h)

        for day_idx, display_code in enumerate(row.codes_display):
            cx = day_start_x + day_idx * day_w
            fill = '#FFFFFF' if idx % 2 == 0 else '#F9FAFA'
            if template == 'highlighted_grid' and row.codes_canonical[day_idx] not in {'D', 'E', 'N', 'F', 'f', 'M', 'OFF', 'O'}:
                fill = rng.choice(['#FF63BD', '#DDF05A', '#67C9E8'])
            if template == 'compact_summary':
                base = row.codes_canonical[day_idx].upper()
                if base == 'D': fill = '#DCEBD7'
                elif base == 'E': fill = '#F6E0D7'
                elif base == 'N': fill = '#DDE5F6'
                elif base in {'OFF', 'O', 'F'}: fill = '#F7DDE3'
            draw.rectangle((cx, row_y, cx + day_w, row_y + row_h), fill=fill, outline='#9A9A9A')
            fitted_text(draw, (cx, row_y, cx + day_w, row_y + row_h), display_code,
                        max_size=11 if len(display_code) <= 2 else 9, min_size=5)
            annotations.append({
                'schedule_id': schedule.schedule_id,
                'template_id': schedule.template_id,
                'row_id': row.row_id,
                'row_index': idx + 1,
                'name': row.name,
                'surname': row.surname,
                'surname_rank': row.surname_rank,
                'surname_population': row.surname_population,
                'surname_hanja_variants': row.surname_hanja_variants,
                'birth_year': row.birth_year,
                'gender': row.gender,
                'group': row.group,
                'day': day_idx + 1,
                'date': f'{schedule.year:04d}-{schedule.month:02d}-{day_idx + 1:02d}',
                'canonical_code': row.codes_canonical[day_idx],
                'display_code': display_code,
                'bbox_px': [cx, row_y, cx + day_w, row_y + row_h],
                'name_bbox_px': list(name_box),
            })

        if summary_cols:
            normalized = [code.upper() for code in row.codes_canonical]
            counts = {
                'D': normalized.count('D'),
                'E': normalized.count('E'),
                'N': normalized.count('N'),
                'OFF': sum(1 for code in normalized if code in {'OFF', 'O', 'F', 'OF'}),
                '연차': sum(1 for code in row.codes_canonical if code in {'연', '연차', '연가', 'AL', 'A/L'}),
            }
            for j, label in enumerate(summary_cols):
                sx = summary_start_x + j * summary_w
                draw.rectangle((sx, row_y, sx + summary_w, row_y + row_h), fill='#FFFFFF', outline='#777777')
                centered_text(draw, (sx, row_y, sx + summary_w, row_y + row_h), str(counts.get(label, 0)), small)

    # Merge-like group labels by painting over internal edges for grouped templates.
    if template in {'grouped_hospital', 'parted_pdf'}:
        for group, (start, end) in group_ranges.items():
            gy1 = body_y + start * row_h
            gy2 = body_y + (end + 1) * row_h
            group_index = int(re.sub(r'\D', '', group) or '1') - 1
            fill = group_palette[group_index % len(group_palette)]
            draw.rectangle((table_x, gy1, table_x + left_group, gy2), fill=fill, outline='#222222', width=2)
            label = ('수속\n' if template == 'parted_pdf' else '') + group
            lines = label.split('\n')
            if len(lines) == 1:
                centered_text(draw, (table_x, gy1, table_x + left_group, gy2), label, bold)
            else:
                total_h = len(lines) * 20
                for li, line in enumerate(lines):
                    box = (table_x, int((gy1 + gy2 - total_h) / 2 + li * 20), table_x + left_group,
                           int((gy1 + gy2 - total_h) / 2 + (li + 1) * 20))
                    centered_text(draw, box, line, bold)

    # Slight paper texture for photographed-table similarity without changing bboxes.
    if rng.random() < 0.65:
        overlay = Image.new('L', image.size, 0)
        pixels = overlay.load()
        for _ in range(max(1000, image.width * image.height // 160)):
            px = rng.randrange(image.width)
            py = rng.randrange(image.height)
            pixels[px, py] = rng.randint(3, 18)
        texture = Image.new('RGB', image.size, '#000000')
        image = Image.composite(texture, image, overlay)
        image = ImageEnhance.Brightness(image).enhance(1.03)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)
    schedule.clean_image_path = str(output_path)
    schedule.image_width = image.width
    schedule.image_height = image.height
    schedule.cell_annotations = annotations


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def apply_basic_style(rng: Any, *, fill: str | None = None, bold: bool = False,
                      color: str = '#111111', size: int = 10,
                      h_align: str = 'center', wrap: bool = True) -> None:
    fmt: dict[str, Any] = {
        'font': {'bold': bold, 'color': color, 'size': size},
        'horizontal_alignment': h_align,
        'vertical_alignment': 'center',
        'wrap_text': wrap,
    }
    if fill:
        fmt['fill'] = fill
    rng.format = fmt


def export_dataset_xlsx(
    schedules: list[ScheduleRecord],
    name_entries: list[NameEntry],
    surname_entries: list[SurnameEntry],
    output_path: Path,
) -> None:
    # Assign stable Excel row/cell addresses before writing ground-truth sheets.
    for s in schedules:
        group_col = s.template_id in {'compact_summary', 'grouped_hospital', 'parted_pdf'}
        for idx, row in enumerate(s.rows):
            row.excel_row = 4 + idx
            row.name_cell = f"{'B' if group_col else 'A'}{row.excel_row}"

    wb = Workbook.create()

    readme = wb.worksheets.add('README')
    readme.merge_cells('A1:H1')
    readme.get_range('A1').values = [['합성 간호사 근무표 데이터셋']]
    apply_basic_style(readme.get_range('A1:H1'), fill='#1F4E78', bold=True, color='#FFFFFF', size=16)
    notes = [
        ['항목', '내용'],
        ['목적', '근무표 OCR/표 구조 인식 학습용 합성 데이터'],
        ['개인정보', '실제 직원 정보가 아닌 합성 이름과 무작위 근무 코드'],
        ['이름 범위', f'{MIN_BIRTH_YEAR}~{MAX_BIRTH_YEAR}년생'],
        ['이름 데이터 주의', '1966~2007년은 공개된 완전한 연도별 순위가 없어 역사적 기준연도에 가장 가까운 이름 풀을 사용'],
        ['성씨', '2015 인구총조사 성씨 순위와 인구수로 가중 추출하며 동일 한글 성씨의 한자 변형은 합산'],
        ['정답지', 'ground_truth_rows 및 ground_truth_cells 시트'],
        ['이미지 좌표', 'ground_truth_cells의 bbox_px는 clean PNG 기준 [x1,y1,x2,y2]'],
        ['코드', 'code_dictionary 시트에 전체 정규 코드와 그룹 수록'],
    ]
    readme.get_range('A3:B11').values = notes
    apply_basic_style(readme.get_range('A3:B3'), fill='#D9EAF7', bold=True)
    readme.get_range('A:A').format.column_width = 22
    readme.get_range('B:B').format.column_width = 75
    readme.get_range('A3:B11').format.row_height = 24

    codes_ws = wb.worksheets.add('code_dictionary')
    codes_ws.get_range('A1:C1').values = [['group', 'canonical_code', 'english_case_mutable']]
    code_rows: list[list[Any]] = []
    for group, codes in SHIFT_CODE_GROUPS.items():
        for code in dedupe(codes):
            code_rows.append([group, code, any('A' <= ch.upper() <= 'Z' for ch in code)])
    codes_ws.get_range_by_indexes(1, 0, len(code_rows), 3).values = code_rows
    apply_basic_style(codes_ws.get_range('A1:C1'), fill='#385723', bold=True, color='#FFFFFF')
    codes_ws.freeze_panes.freeze_rows(1)
    codes_ws.get_range('A:C').format.column_width = 22

    names_ws = wb.worksheets.add('names_by_birth_year')
    name_headers = ['birth_year', 'gender', 'rank', 'given_name', 'source_year', 'source_method', 'source_url']
    names_ws.get_range('A1:G1').values = [name_headers]
    name_rows = [[
        e.birth_year, e.gender, e.rank, e.given_name, e.source_year, e.source_method, e.source_url
    ] for e in name_entries]
    names_ws.get_range_by_indexes(1, 0, len(name_rows), len(name_headers)).values = name_rows
    apply_basic_style(names_ws.get_range('A1:G1'), fill='#8064A2', bold=True, color='#FFFFFF')
    names_ws.freeze_panes.freeze_rows(1)
    names_ws.get_range('A:F').format.column_width = 18
    names_ws.get_range('G:G').format.column_width = 48

    surname_ws = wb.worksheets.add('surname_dictionary')
    surname_headers = [
        'rank', 'surname', 'hanja', 'population', 'source_year', 'source_method',
        'source_url', 'aggregated_population', 'aggregated_best_rank',
        'hanja_variants', 'sampling_weight_power_0_75'
    ]
    surname_ws.get_range_by_indexes(0, 0, 1, len(surname_headers)).values = [surname_headers]
    surname_pool = aggregate_surnames(surname_entries)
    surname_by_text = {item.surname: item for item in surname_pool}
    surname_rows = []
    for entry in surname_entries:
        aggregate = surname_by_text[entry.surname]
        surname_rows.append([
            entry.rank, entry.surname, entry.hanja, entry.population, entry.source_year,
            entry.source_method, entry.source_url, aggregate.population,
            aggregate.best_rank, aggregate.hanja_variants,
            round(aggregate.population ** 0.75, 6),
        ])
    surname_ws.get_range_by_indexes(1, 0, len(surname_rows), len(surname_headers)).values = surname_rows
    apply_basic_style(
        surname_ws.get_range_by_indexes(0, 0, 1, len(surname_headers)),
        fill='#7F6000', bold=True, color='#FFFFFF'
    )
    surname_ws.freeze_panes.freeze_rows(1)
    surname_ws.get_range('A:F').format.column_width = 18
    surname_ws.get_range('G:G').format.column_width = 48
    surname_ws.get_range('H:K').format.column_width = 22

    manifest_ws = wb.worksheets.add('manifest')
    manifest_headers = ['schedule_id', 'template_id', 'year', 'month', 'day_count', 'people_count',
                        'sheet_name', 'clean_image_path', 'image_width', 'image_height']
    manifest_ws.get_range('A1:J1').values = [manifest_headers]
    manifest_rows = [[
        s.schedule_id, s.template_id, s.year, s.month, s.day_count, len(s.rows), s.sheet_name,
        s.clean_image_path, s.image_width, s.image_height
    ] for s in schedules]
    manifest_ws.get_range_by_indexes(1, 0, len(manifest_rows), len(manifest_headers)).values = manifest_rows
    apply_basic_style(manifest_ws.get_range('A1:J1'), fill='#1F4E78', bold=True, color='#FFFFFF')
    manifest_ws.freeze_panes.freeze_rows(1)
    manifest_ws.get_range('A:J').format.column_width = 20

    row_gt_ws = wb.worksheets.add('ground_truth_rows')
    row_headers = ['schedule_id', 'template_id', 'sheet_name', 'row_id', 'row_index', 'excel_row',
                   'group', 'name', 'surname', 'surname_rank', 'surname_population',
                   'surname_hanja_variants', 'surname_source_method', 'surname_source_url',
                   'given_name', 'birth_year', 'gender', 'day_count',
                   'codes_canonical_json', 'codes_display_json', 'codes_canonical_joined',
                   'codes_display_joined', 'name_cell']
    row_gt_ws.get_range_by_indexes(0, 0, 1, len(row_headers)).values = [row_headers]
    row_data: list[list[Any]] = []
    for s in schedules:
        for row_index, row in enumerate(s.rows, start=1):
            row_data.append([
                s.schedule_id, s.template_id, s.sheet_name, row.row_id, row_index, row.excel_row,
                row.group, row.name, row.surname, row.surname_rank, row.surname_population,
                row.surname_hanja_variants, row.surname_source_method, row.surname_source_url,
                row.given_name, row.birth_year, row.gender,
                s.day_count, json.dumps(row.codes_canonical, ensure_ascii=False),
                json.dumps(row.codes_display, ensure_ascii=False),
                '|'.join(row.codes_canonical), '|'.join(row.codes_display), row.name_cell,
            ])
    row_gt_ws.get_range_by_indexes(1, 0, len(row_data), len(row_headers)).values = row_data
    apply_basic_style(row_gt_ws.get_range_by_indexes(0, 0, 1, len(row_headers)), fill='#C65911', bold=True, color='#FFFFFF')
    row_gt_ws.freeze_panes.freeze_rows(1)
    row_gt_ws.get_range('A:R').format.column_width = 17
    row_gt_ws.get_range('S:W').format.column_width = 42

    cell_gt_ws = wb.worksheets.add('ground_truth_cells')
    cell_headers = ['schedule_id', 'template_id', 'row_id', 'row_index', 'name', 'surname',
                    'surname_rank', 'surname_population', 'birth_year',
                    'gender', 'group', 'day', 'date', 'canonical_code', 'display_code',
                    'excel_cell', 'bbox_px_json', 'name_bbox_px_json', 'image_path']
    cell_gt_ws.get_range_by_indexes(0, 0, 1, len(cell_headers)).values = [cell_headers]
    cell_data: list[list[Any]] = []
    for s in schedules:
        for annotation in s.cell_annotations:
            row = next(r for r in s.rows if r.row_id == annotation['row_id'])
            day = annotation['day']
            excel_cell = f'{excel_column_name(schedule_day_start_col_1based(s) + day - 1)}{row.excel_row}'
            cell_data.append([
                annotation['schedule_id'], annotation['template_id'], annotation['row_id'],
                annotation['row_index'], annotation['name'], annotation['surname'],
                annotation['surname_rank'], annotation['surname_population'],
                annotation['birth_year'], annotation['gender'], annotation['group'],
                day, annotation['date'],
                annotation['canonical_code'], annotation['display_code'], excel_cell,
                json.dumps(annotation['bbox_px']), json.dumps(annotation['name_bbox_px']),
                s.clean_image_path,
            ])
    cell_gt_ws.get_range_by_indexes(1, 0, len(cell_data), len(cell_headers)).values = cell_data
    apply_basic_style(cell_gt_ws.get_range_by_indexes(0, 0, 1, len(cell_headers)), fill='#BF9000', bold=True, color='#FFFFFF')
    cell_gt_ws.freeze_panes.freeze_rows(1)
    cell_gt_ws.get_range('A:S').format.column_width = 18

    sources_ws = wb.worksheets.add('sources')
    source_rows = [
        ['source_id', 'purpose', 'url_or_reference', 'note'],
        ['historical_names', '1968/1978/1988/1998/2008 기준 인기 이름', SOURCE_HISTORY,
         '1966~2007 각 연도는 가장 가까운 기준연도 풀로 매핑'],
        ['annual_names', '2008년 이후 선택적 실시간 스크래핑', SOURCE_BABY_NAME.format(year='{year}', page='{page}'),
         '사이트 구조 변경 또는 이용정책에 따라 실패할 수 있어 fallback 포함'],
        ['surname_official', '2015 성씨ㆍ본관별 인구 공식 통계표', SOURCE_SURNAME_KOSIS,
         'KOSIS 인구총조사 DT_1IN15SD'],
        ['surname_scraper', '2015 성씨 순위 전체 HTML 스크래핑', SOURCE_SURNAME_MIRROR,
         '온라인이면 공개된 532개 순위 행을 파싱, 실패하면 내장 상위 100개 fallback'],
        ['template_reference', '병원식 파트/색상/월간 표 레이아웃 참고', SOURCE_TEMPLATE_PDF,
         '사용자가 제공한 1페이지 근무 편성표'],
    ]
    sources_ws.get_range('A1:D6').values = source_rows
    apply_basic_style(sources_ws.get_range('A1:D1'), fill='#5B9BD5', bold=True, color='#FFFFFF')
    sources_ws.get_range('A:D').format.column_width = 38

    # Schedule sheets.
    for s in schedules:
        ws = wb.worksheets.add(s.sheet_name)
        group_col = s.template_id in {'compact_summary', 'grouped_hospital', 'parted_pdf'}
        start_col = 0
        day_col = 2 if group_col else 1
        total_cols = day_col + s.day_count
        ws.merge_cells(f'A1:{excel_column_name(total_cols)}1')
        ws.get_range('A1').values = [[s.title]]
        title_fill = '#F8B800' if s.template_id in {'compact_summary', 'grouped_hospital'} else '#D9EAD3'
        apply_basic_style(ws.get_range_by_indexes(0, 0, 1, total_cols), fill=title_fill, bold=True, size=14)
        header_row = 1
        if group_col:
            ws.merge_cells('A2:A3')
            ws.get_range('A2').values = [['구분']]
            ws.merge_cells('B2:B3')
            ws.get_range('B2').values = [['성명']]
        else:
            ws.merge_cells('A2:A3')
            ws.get_range('A2').values = [['성명']]
        for day in range(1, s.day_count + 1):
            col = day_col + day - 1
            ws.get_range_by_indexes(header_row, col, 1, 1).values = [[day]]
            ws.get_range_by_indexes(header_row + 1, col, 1, 1).values = [[s.weekdays[day - 1]]]
        apply_basic_style(ws.get_range_by_indexes(header_row, 0, 2, total_cols), fill='#F2F2F2', bold=True)

        body_start = header_row + 2
        for idx, row in enumerate(s.rows):
            excel_row_1based = row.excel_row
            if group_col:
                ws.get_range_by_indexes(body_start + idx, 0, 1, 1).values = [[row.group]]
                ws.get_range_by_indexes(body_start + idx, 1, 1, 1).values = [[row.name]]
                row.name_cell = f'B{excel_row_1based}'
            else:
                ws.get_range_by_indexes(body_start + idx, 0, 1, 1).values = [[row.name]]
                row.name_cell = f'A{excel_row_1based}'
            ws.get_range_by_indexes(body_start + idx, day_col, 1, s.day_count).values = [row.codes_display]
            fill = '#FFFFFF' if idx % 2 == 0 else '#F8FAFA'
            apply_basic_style(ws.get_range_by_indexes(body_start + idx, 0, 1, total_cols), fill=fill, size=9)
        ws.freeze_panes.freeze_rows(body_start)
        ws.freeze_panes.freeze_columns(day_col)
        ws.get_range_by_indexes(0, 0, body_start + len(s.rows), 1).format.column_width = 15 if not group_col else 12
        if group_col:
            ws.get_range_by_indexes(0, 1, body_start + len(s.rows), 1).format.column_width = 14
        ws.get_range_by_indexes(0, day_col, body_start + len(s.rows), s.day_count).format.column_width = 5.2
        ws.get_range_by_indexes(0, 0, body_start + len(s.rows), total_cols).format.row_height = 21

    output_path.parent.mkdir(parents=True, exist_ok=True)
    SpreadsheetFile.export_xlsx(wb).save(str(output_path))


def export_annotations(schedules: list[ScheduleRecord], output_dir: Path) -> None:
    rows_jsonl: list[dict[str, Any]] = []
    cells_jsonl: list[dict[str, Any]] = []
    for schedule in schedules:
        for row_index, row in enumerate(schedule.rows, start=1):
            rows_jsonl.append({
                'schedule_id': schedule.schedule_id,
                'template_id': schedule.template_id,
                'sheet_name': schedule.sheet_name,
                'row_id': row.row_id,
                'row_index': row_index,
                'excel_row': row.excel_row,
                'group': row.group,
                'name': row.name,
                'surname': row.surname,
                'surname_rank': row.surname_rank,
                'surname_population': row.surname_population,
                'surname_hanja_variants': row.surname_hanja_variants,
                'surname_source_method': row.surname_source_method,
                'surname_source_url': row.surname_source_url,
                'given_name': row.given_name,
                'birth_year': row.birth_year,
                'gender': row.gender,
                'day_count': schedule.day_count,
                'codes_canonical': row.codes_canonical,
                'codes_display': row.codes_display,
                'codes_canonical_joined': '|'.join(row.codes_canonical),
                'codes_display_joined': '|'.join(row.codes_display),
                'name_cell': row.name_cell,
                'image_path': schedule.clean_image_path,
            })
        for annotation in schedule.cell_annotations:
            row = next(r for r in schedule.rows if r.row_id == annotation['row_id'])
            annotation = dict(annotation)
            annotation['excel_cell'] = f'{excel_column_name(schedule_day_start_col_1based(schedule) + annotation["day"] - 1)}{row.excel_row}'
            annotation['image_path'] = schedule.clean_image_path
            cells_jsonl.append(annotation)

    write_jsonl(output_dir / 'annotations' / 'rows.jsonl', rows_jsonl)
    write_jsonl(output_dir / 'annotations' / 'cells.jsonl', cells_jsonl)

    row_csv = []
    for item in rows_jsonl:
        row_csv.append({**item,
                        'codes_canonical': json.dumps(item['codes_canonical'], ensure_ascii=False),
                        'codes_display': json.dumps(item['codes_display'], ensure_ascii=False)})
    write_csv(output_dir / 'annotations' / 'rows.csv', row_csv)

    cell_csv = []
    for item in cells_jsonl:
        cell_csv.append({**item,
                         'bbox_px': json.dumps(item['bbox_px']),
                         'name_bbox_px': json.dumps(item['name_bbox_px'])})
    write_csv(output_dir / 'annotations' / 'cells.csv', cell_csv)

    manifest = {
        'dataset_version': '1.1',
        'year': CURRENT_YEAR,
        'schedule_count': len(schedules),
        'all_canonical_codes': ALL_SHIFT_CODES,
        'schedules': [{
            'schedule_id': s.schedule_id,
            'template_id': s.template_id,
            'year': s.year,
            'month': s.month,
            'day_count': s.day_count,
            'people_count': len(s.rows),
            'sheet_name': s.sheet_name,
            'image_path': s.clean_image_path,
            'image_size': [s.image_width, s.image_height],
        } for s in schedules],
    }
    (output_dir / 'annotations' / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
    )


def export_dictionaries(
    name_entries: list[NameEntry],
    surname_entries: list[SurnameEntry],
    output_dir: Path,
) -> None:
    name_rows = [asdict(entry) for entry in name_entries]
    write_csv(output_dir / 'name_dictionary.csv', name_rows)

    surname_rows = [asdict(entry) for entry in surname_entries]
    write_csv(output_dir / 'surname_dictionary.csv', surname_rows)

    aggregated_rows = [asdict(entry) for entry in aggregate_surnames(surname_entries)]
    write_csv(output_dir / 'surname_sampling_pool.csv', aggregated_rows)

    code_rows = []
    for group, codes in SHIFT_CODE_GROUPS.items():
        for code in dedupe(codes):
            code_rows.append({
                'group': group,
                'canonical_code': code,
                'english_case_mutable': any('A' <= ch.upper() <= 'Z' for ch in code),
            })
    write_csv(output_dir / 'code_dictionary.csv', code_rows)


def generate_dataset(
    config: GeneratorConfig,
    force_template_cycle: bool = False,
) -> tuple[list[ScheduleRecord], list[NameEntry], Path]:
    rng = random.Random(config.seed)
    output_dir = Path(config.output_dir).resolve()
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'annotations').mkdir(parents=True, exist_ok=True)

    name_entries = build_name_dictionary(
        config.min_birth_year,
        config.max_birth_year,
        scrape_recent=config.scrape_recent_names,
        recent_top_n=config.recent_name_top_n,
    )
    surname_entries = build_surname_dictionary(
        scrape=config.scrape_surnames,
        max_rank=config.surname_max_rank,
        min_population=config.surname_min_population,
    )
    surname_pool = aggregate_surnames(surname_entries)

    schedules: list[ScheduleRecord] = []
    for i in range(1, config.count + 1):
        forced = config.template_ids[(i - 1) % len(config.template_ids)] if force_template_cycle else None
        schedule = generate_schedule(
            i,
            config,
            rng,
            name_entries,
            surname_pool,
            forced_template=forced,
        )
        schedules.append(schedule)

    if config.ensure_all_codes:
        ensure_code_coverage(schedules, rng, config.case_mutation_probability)

    for schedule in schedules:
        image_path = output_dir / 'images' / f'{schedule.schedule_id}_{schedule.template_id}.png'
        render_schedule_png(schedule, image_path, rng)
        schedule.clean_image_path = str(Path('images') / image_path.name)

    xlsx_path = output_dir / 'synthetic_shift_dataset.xlsx'
    export_dataset_xlsx(schedules, name_entries, surname_entries, xlsx_path)
    export_annotations(schedules, output_dir)
    export_dictionaries(name_entries, surname_entries, output_dir)
    return schedules, name_entries, xlsx_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate anonymous nurse shift schedule datasets.')
    parser.add_argument('--count', type=int, default=5, help='Number of schedules to generate')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--seed', type=int, default=20260723)
    parser.add_argument('--min-people', type=int, default=18)
    parser.add_argument('--max-people', type=int, default=32)
    parser.add_argument('--config', help='Optional JSON config path')
    parser.add_argument('--scrape-recent-names', action='store_true')
    parser.add_argument('--no-scrape-surnames', action='store_true', help='Use bundled top-100 surname fallback only')
    parser.add_argument('--surname-max-rank', type=int, default=532)
    parser.add_argument('--surname-min-population', type=int, default=5)
    parser.add_argument('--surname-weight-power', type=float, default=0.75)
    parser.add_argument('--cycle-templates', action='store_true', help='Cycle through all templates deterministically')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.config:
        config = GeneratorConfig.from_json(args.config)
    else:
        config = GeneratorConfig(
            count=args.count,
            output_dir=args.output_dir,
            seed=args.seed,
            min_people=args.min_people,
            max_people=args.max_people,
            scrape_recent_names=args.scrape_recent_names,
            scrape_surnames=not args.no_scrape_surnames,
            surname_max_rank=args.surname_max_rank,
            surname_min_population=args.surname_min_population,
            surname_weight_power=args.surname_weight_power,
        )
    if config.count < 1:
        raise ValueError('count must be at least 1')
    if config.min_people < 1 or config.max_people < config.min_people:
        raise ValueError('invalid people range')
    schedules, names, xlsx_path = generate_dataset(config, force_template_cycle=args.cycle_templates)
    print(f'Generated {len(schedules)} schedules')
    print(f'Name dictionary rows: {len(names)}')
    print('Surname source: online 2015 ranking scrape or bundled fallback')
    print(f'Excel: {xlsx_path}')
    print(f'Output directory: {Path(config.output_dir).resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
