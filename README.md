# 합성 간호사 근무표 데이터셋 생성기

사진으로 촬영된 근무표 OCR과 표 구조 인식 학습에 사용할 합성 데이터를 만듭니다.

## 생성되는 파일

```text
output/
├── synthetic_shift_dataset.xlsx
├── name_dictionary.csv
├── surname_dictionary.csv
├── surname_sampling_pool.csv
├── code_dictionary.csv
├── images/
│   ├── schedule_0001_....png
│   └── ...
└── annotations/
    ├── manifest.json
    ├── rows.jsonl
    ├── rows.csv
    ├── cells.jsonl
    └── cells.csv
```

Excel 통합 문서에는 `README`, 코드·이름·성씨 사전, manifest, 행·셀 정답지, 생성된 근무표별 시트와 출처 시트가 포함됩니다.

## 실행

```bash
python generate_dataset.py --count 20 --output-dir output --seed 20260723 --cycle-templates
```

설정 파일 사용:

```bash
python generate_dataset.py --config config.example.json --cycle-templates
```

## 테스트

```bash
python -m unittest discover -s tests -v
```

## 이름 데이터의 중요한 제한

2026년 기준 정년 60세를 가정하여 `1966~2007년생` 이름 풀을 만듭니다.

공개 웹에서 확인 가능한 완전한 연도별 이름 순위는 주로 2008년 이후입니다. 따라서 1966~2007년은 1968·1978·1988·1998·2008 역사적 기준연도의 상위 이름을 사용하고, 가장 가까운 기준연도에 매핑합니다. 모든 이름 행에는 `source_year`, `source_method`, `source_url`을 기록하여 특정 연도의 정확한 순위로 오해하지 않게 했습니다.

`--scrape-recent-names`를 켜면 2008년 이후 지원되는 공개 연도별 페이지를 선택적으로 읽습니다. 사이트 구조나 이용정책이 달라질 수 있으므로 기본값은 꺼짐이며, 실패하면 내장 데이터로 돌아갑니다.


## 성씨 순위와 랜덤 결합

기본 실행에서는 2015 인구총조사 성씨 순위 전체를 온라인으로 읽습니다.

- 공식 통계표: KOSIS `DT_1IN15SD`
- HTML 순위 파싱: `https://www.rootsinfo.co.kr/info/roots/table_sung15.php`
- 공개 페이지에 있는 532개 순위 행을 가능한 범위까지 수집
- 동일한 한글 성씨에 한자가 여러 개인 경우 인구수를 합산
- 합산 인구수의 `0.75`제곱을 가중치로 사용해 흔한 성이 더 자주 나오되 희귀 성도 학습 데이터에 등장할 수 있게 함
- 수집 실패 시 내장된 2015년 상위 100개 성씨 순위로 자동 대체

생성되는 전체 이름은 항상 다음 구조입니다.

```text
성씨 + 이름
김 + 서연 → 김서연
남궁 + 민준 → 남궁민준
```

성씨 출처와 순위는 `surname_dictionary.csv`, `surname_sampling_pool.csv`,
Excel의 `surname_dictionary`, `ground_truth_rows`에 저장됩니다.

온라인 수집을 끄고 내장 사전만 사용하려면:

```bash
python generate_dataset.py --count 20 --no-scrape-surnames
```

순위 범위와 희귀도는 다음처럼 조정합니다.

```bash
python generate_dataset.py \
  --count 100 \
  --surname-max-rank 532 \
  --surname-min-population 5 \
  --surname-weight-power 0.75
```

## 템플릿

1. `compact_summary` — 우측 집계 열이 있는 압축형
2. `clean_grid` — 이름과 날짜가 단순한 흰색 격자형
3. `highlighted_grid` — 일부 코드가 형광 표시된 유형
4. `grouped_hospital` — 제목·공지·그룹·집계가 있는 병원식
5. `parted_pdf` — 파트별 병합 구역과 색상 이름 칸이 있는 유형

업로드한 PDF 1페이지의 파트별 병합 영역, 날짜/요일 2단 헤더, `A53`, `L60`, `D12`, `주`, `연`, `대`, `예` 등 레이아웃을 `parted_pdf` 유형 설계에 참고했습니다.

## 코드 처리

- 요청된 전체 정규 코드를 `code_dictionary`에 저장합니다.
- 배치에 모든 정규 코드가 최소 1회 들어가도록 할 수 있습니다.
- 영어 코드에는 글자별 대소문자 변형을 적용합니다.
- 정답에는 `canonical_code`와 실제 표시값 `display_code`를 모두 기록합니다.
- 한국어, 숫자, `/`, `+`, `⁺`, `—`는 그대로 보존합니다.

## 학습 정답

행 정답에는 전체 이름, 성씨, 성씨 순위·인구수·한자 변형·출처, 이름, 출생연도, 성별, 파트, Excel 행 번호, 코드 배열과 순서 문자열을 저장합니다. 셀 정답에는 날짜, 정규 코드, 표시 코드, Excel 셀 주소, PNG 기준 `bbox_px`, 이름 셀 `name_bbox_px`를 저장합니다.

## 개인정보

생성되는 이름은 흔한 성씨와 시대별 이름 풀을 무작위로 결합한 합성값입니다. 실제 병원명이나 실제 직원 데이터는 사용하지 않습니다.

## 대량 생성

```bash
python generate_dataset.py --count 500 --output-dir dataset_500 --seed 42
```

Excel에 수백 개 시트를 넣으면 파일이 커질 수 있습니다. 대규모 학습에서는 PNG와 JSONL을 주 데이터로 사용하고 Excel은 검수용으로 사용하는 방식을 권장합니다.
