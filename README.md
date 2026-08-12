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

Excel 통합 문서에는 `README`, 코드·이름·성씨 사전, manifest, 행·셀 정답지, 생성된 근무표별 시트와 출처 시트가 포함됩니다. 통합 문서는 항상 보관됩니다.

각 PNG는 별도 그림 코드로 다시 그리지 않습니다. 완성된 Excel 근무표 시트의 제목부터 마지막 근무 행·집계 열까지를 `2x` 해상도로 직접 렌더링하므로 Excel과 동일한 글꼴, 중앙 정렬, 병합 셀, 색상을 유지합니다. Excel의 열 문자와 행 번호는 PNG에 포함되지 않습니다.

## 실행

```bash
python generate_dataset.py --count 20 --output-dir output --seed 20260723 --cycle-templates
```

설정 파일 사용:

```bash
python generate_dataset.py --config config.example.json --cycle-templates
```

Excel/PNG 렌더링에는 Node.js와 Codex/ChatGPT 환경의 `@oai/artifact-tool`이 필요합니다. 비표준 런타임에서는 다음 환경 변수로 번들 경로를 지정할 수 있습니다.

```text
ARTIFACT_TOOL_NODE=<node 실행 파일>
ARTIFACT_TOOL_NODE_MODULES=<@oai/artifact-tool이 들어 있는 node_modules>
```

## 여러 장의 페이지 표시

근무표를 2장 이상 생성하면 각 Excel 시트의 제목과 PNG 우측 가장자리에
`페이지 1/20` 형식의 번호가 자동으로 표시됩니다. `manifest`에도
`page_number`, `page_count`, `page_label`이 저장되어 출력물이 섞여도 순서를
다시 맞출 수 있습니다.

페이지 표시를 끄려면:

```bash
python generate_dataset.py --count 20 --no-page-numbers
```

JSON 설정에서는 `"show_page_numbers": false`로 끌 수 있습니다.

## 대량 인쇄 배치와 정답지

인쇄용 묶음은 기본적으로 근무표를 25장씩 나눠 렌더링합니다. 중간 통합 Excel은
만들지 않으며, PNG와 100페이지 단위 PDF를 생성한 뒤 페이지·행·셀 정답과
검수용 Excel 인덱스를 함께 저장합니다.

```bash
python generate_print_batch.py --count 1000 --output-dir outputs/print_batch_1000
```

생성 폴더에는 `images/`, `pdf/`, `annotations/`, `print_manifest.csv`,
`print_manifest.json`, `training_answer_index_1000.xlsx`, `PRINT_GUIDE_KO.txt`가
포함됩니다. 메모리에 맞춰 렌더링 묶음 크기를 바꾸려면
`--render-chunk-size 10`, PDF 한 권의 페이지 수를 바꾸려면
`--pages-per-pdf 50`처럼 지정합니다.

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
- `보건휴가`, `경조사`, `병가`, `출산휴가`, `육아휴직`, `노조휴가`, `휴직`과 축약형은 일반 한국어 코드보다 낮은 `0.35` 가중치로 무작위 추출합니다.
- 전체 코드 포함 설정을 켜면 희귀 코드도 생성 묶음 전체에 최소 1회 포함됩니다.
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

Excel에 수백 개 시트를 넣으면 파일이 커질 수 있습니다. 통합 Excel은 항상 생성되며, 대규모 학습에서는 PNG와 JSONL을 주 데이터로 사용하고 Excel은 검수용으로 사용하는 방식을 권장합니다.

## OCR 학습 파이프라인

데스크탑의 기본 학습 저장 루트는 `D:\harudam_model`입니다. 경로 옵션을 생략하면 합성 데이터와 shard는 `D:\harudam_model\training_dataset`, 모델별 checkpoint와 로그는 `D:\harudam_model\runs\<model>_<phase>`에 저장됩니다. 다른 위치가 필요할 때만 `--storage-root` 또는 개별 경로 옵션을 명시하세요.

학습용 전체 데이터는 다음 명령으로 생성합니다. 기본값은 in-distribution 10,000장, 기존 5개 템플릿과 겹치지 않는 OOD layout 200장, 실제 촬영 대상 300 schedule/1,000장입니다. 노트북에서는 `--count 25 --ood-count 4 --skip-parquet` 정도로만 smoke test하고 전체 생성·학습은 CUDA 작업 환경에서 실행하세요.

```bash
python generate_training_dataset.py \
  --count 10000 \
  --ood-count 200 \
  --render-chunk-size 25
```

이 명령은 어떤 PNG나 crop도 만들기 전에 `splits/master_split.jsonl`을 만들고 SHA-256을 잠급니다. 모든 합성 원본·실제 촬영본·등록 결과·증강 recipe·recognition crop은 `schedule_id`로 이 split을 상속합니다. Train만 `cv_fold=0|1|2`이고 Validation/Test/OOD는 항상 `-1`입니다. loader는 자체 split을 만들지 않으며 unknown ID, split mismatch, Validation/Test/OOD 학습 유입, Test/OOD 기반 threshold·양자화·경로 선택을 예외로 중단합니다.

학습용 Excel 청크에는 PNG와 동일한 근무표 시트, README, manifest만 넣습니다. 수만 행의 정답과 동일한 사전을 매 청크에 중복하지 않으며 canonical 정답은 `annotations/`의 CSV/JSONL과 `shards/`의 Parquet에 한 번만 저장합니다. 일반 생성과 인쇄용 Excel의 정답 시트 구조는 그대로 유지됩니다.

대량 결과는 다음 구조입니다.

```text
training_dataset/
├── splits/master_split.jsonl
├── splits/master_split.manifest.json
├── workbooks/synthetic_shift_dataset_*.xlsx
├── images/*.png
├── annotations/
│   ├── rows.{csv,jsonl}
│   ├── cells.{csv,jsonl}
│   ├── objects.{csv,jsonl}
│   ├── charset_coverage.json
│   └── manifest.json
└── shards/
    ├── train-*.parquet
    ├── validation-*.parquet
    ├── test-*.parquet
    ├── ood_layout-*.parquet
    ├── image_index.parquet
    └── recognition_index.parquet
```

`objects`에는 제목, 날짜·요일, 그룹·이름, 근무 코드, 집계값이 모두 들어갑니다. `bbox_px`는 `cell_polygon`의 AABB이고, `text_polygon`은 같은 text object의 전체 glyph를 표현하는 하나의 convex/min-area quadrilateral입니다. 렌더 layout bounds를 우선 쓰고 2px 검증에 실패하면 glyph mask, 마지막으로 text/no-text 렌더 차분을 사용할 수 있습니다. 개별 획 contour는 정답으로 저장하지 않습니다.

CSV/JSONL/Parquet 일치 여부와 Unicode/사전 coverage는 다음처럼 다시 검사합니다.

```bash
python verify_annotations.py \
  --annotations-dir training_dataset/annotations \
  --shard-dir training_dataset/shards

python check_charset.py \
  --objects training_dataset/annotations/objects.jsonl \
  --output training_dataset/annotations/charset_coverage.recheck.json
```

Parquet manifest는 `training_shards_v2`이며 split/CV, 표시·정규 코드, bbox와 두 polygon, glyph 검증 오차, visibility/ignore, 이미지 크기까지 포함한 checksum으로 CSV·JSONL·Parquet의 학습 의미가 같은지 확인합니다. 기존 v1 shard reader도 유지됩니다.

고정 CTC 사전은 `data/korean_charset_v1.txt`입니다. `@range` 지시자는 고정된 한글 codepoint 순서를 뜻하며 dataset 문자에서 사전을 다시 만들지 않습니다. 문자 순서는 CTC class index이므로 기존 버전을 수정하지 말고 새 버전 파일과 모델을 함께 만들어야 합니다. transcription은 NFC `display_text`이며 `⁺`, `+`, `/`, `—`, `-`와 영문 대소문자를 보존합니다. `canonical_code` 변환은 OCR 이후 `shift_ocr.canonicalize.CodeCanonicalizer`에서 수행합니다.

### 실제 사진 등록

```bash
python register_real_photos.py \
  --reference training_dataset/images/schedule_0001_clean_grid.png \
  --photo captures/schedule_0001/front.jpg \
  --objects captures/schedule_0001/source_objects.jsonl \
  --master-split training_dataset/splits/master_split.jsonl \
  --photo-path-in-dataset captures/schedule_0001/front.jpg \
  --output captures/schedule_0001/front.registration.json
```

EXIF 방향을 먼저 적용하고 원본과 사진의 long side를 2,400px로 정규화합니다. SIFT+FLANN, 필요 시 AKAZE+BF, RANSAC Homography 순으로 진행하며 ECC는 기하학적으로 유효한 초기 Homography의 보정에만 사용합니다. 승인에는 match/inlier 수뿐 아니라 양쪽 convex-hull coverage, 공간 구역 분포, symmetric transfer error와 변환 quad의 convexity·각도·면적·종횡비가 포함됩니다. 강한 partial crop은 `--partial`을 사용해 공통 가시 영역 기준으로 평가합니다. visibility 60% 이상은 정상, 20~60%는 ignore, 20% 미만은 제거합니다.

### 모델 학습과 평가

별도 CUDA 환경에 `requirements-training.txt`를 설치합니다. artifact-tool은 데이터 생성에만 사용하며 학습 코드에는 의존하지 않습니다.

```bash
python train_models.py --model table \
  --image-size 1280 \
  --batch-size 8 \
  --effective-batch-size 32

python train_models.py --model recognizer \
  --phase real_finetune \
  --resume D:\harudam_model\runs\recognizer_pretrain\best.pt \
  --epochs 10 --resume-lr-policy reset \
  --storage-root D:\harudam_model
```

10,000장 본 학습은 위와 같이 `--shard-dir`을 사용합니다. 이 경로는 image/recognition index만 메모리에 두고 annotation은 worker별로 필요한 Parquet row group만 읽습니다. `--objects`는 작은 디버그 데이터와 구형 shard가 없는 경우를 위한 호환 경로이며 `--shard-dir`과 동시에 지정할 수 없습니다.

세 모델은 독립 학습됩니다.

- DBNet: MobileNetV3-FPN, text polygon Hmean@IoU 0.5로 best 선택
- recognizer: MobileNetV3-BiLSTM-CTC, cell exact accuracy 우선·동률 CER로 best 선택
- table: MobileNetV3-FPN, `0.6 × cell polygon F1 + 0.4 × row accuracy`; dense head는 1/4 feature, 선택적 attention은 1/16 feature

table decode는 기본 2,048 후보이며 1,600 미만 설정을 거부합니다. center Gaussian과 corner offset으로 cell quad를 복원하고, quad IoU 0.5의 1:1 매칭으로 cell polygon F1을 계산합니다. row/column embedding은 decode와 관계 정확도 평가에 모두 사용합니다. CUDA에서는 실제 DataLoader 배치로 1~2 iteration을 실행해 peak VRAM을 측정하고, OOM이면 1,280px 해상도를 유지한 채 physical batch부터 낮춥니다. Windows의 `num_workers`는 자동으로 보수적인 값이 선택되며 `--num-workers`로 고정할 수 있습니다. 선택값과 변경 이유는 `runtime_settings.json`에 남습니다.

checkpoint는 best/last/주기 파일과 optimizer·scheduler·AMP scaler·RNG 상태를 저장해 CUDA resume를 지원합니다. resume 시 `--epochs`는 추가 epoch 수이며, `--resume-lr-policy restore|reset`으로 checkpoint LR을 이어가거나 새 fine-tuning LR로 명시적으로 초기화합니다. phase 또는 Validation selection scope가 바뀌면 이전 best/history는 자동 초기화되어 새 scope의 첫 평가가 `best.pt`를 만듭니다. epoch별 train/validation loss, 실제 group learning rate, GPU VRAM(MB), 시간, 모델별 Validation 지표와 best 갱신 여부는 `training_history.csv`, `training_history.json`, `training_manifest.json`에 기록됩니다. grouped 3-fold 비교는 `--cv-fold 0|1|2`로 Train 안에서만 실행합니다.

증강 이미지는 저장하지 않습니다. Dataset은 공유 absolute epoch를 사용해 같은 epoch·seed에서는 재현되면서 다음 epoch에는 다른 recipe/crop jitter를 만들며, Windows persistent worker와 checkpoint resume에서도 동일한 절대 epoch 규칙을 따릅니다.

최종 지표는 `evaluate_models.py metrics`로 cell exact, CER, row exact, full-schedule exact, text detection Hmean, cell polygon F1을 기록합니다. Test와 OOD는 schedule 단위 2,000회 bootstrap 95% CI를 포함하고 별도 보고합니다. A/B/C/D 경로 선택은 `select-route`에서 real Validation만 허용하며 Test/OOD 입력은 거부합니다.

### ONNX와 모바일 profile

```bash
python export_mobile.py --model recognizer \
  --checkpoint runs/recognizer_real/best.pt \
  --output-dir exports/recognizer

python quantize_models.py --model recognizer \
  --input exports/recognizer/recognizer_dynamic_batch.onnx \
  --output-dir exports/recognizer/quantized
```

recognizer는 CPU/XNNPACK용 dynamic-batch와 NNAPI/CoreML 비교용 fixed batch/fixed width를 모두 export합니다. width 160은 batch 1/4/8/16, 320은 1/2/4/8, 640은 1/2/4입니다. DBNet/table도 dynamic/fixed shape를 비교합니다. CNN은 static INT8 QDQ·FP16·FP32, BiLSTM recognizer는 dynamic INT8·CNN-only static INT8·FP16·FP32를 따로 평가합니다. ORT usability checker 결과와 실제 Android Validation benchmark를 `select_mobile_profile.py`에 넣은 뒤 선택된 모델 profile과 batch=1 fallback만 앱 assets에 포함합니다.

앱 baseline은 새 학습 전에 `benchmark_existing_app.py`로 동일 real Validation의 기존 `DBNet→SVTR`, `SLANet→SVTR`, DBNet clustering fallback을 저장합니다. 모델·앱 commit·기기·OS·EP·입력 해상도, p50/p95 latency와 peak memory를 이후 모든 실험과 비교합니다.
