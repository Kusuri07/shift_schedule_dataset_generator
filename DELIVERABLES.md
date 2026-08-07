# Deliverables

- `generate_dataset.py`: configurable schedule generator with online surname-rank scraping
- `render_workbook.mjs`: artifact-tool workbook builder and 2x schedule-sheet PNG renderer
- `config.example.json`: configuration example
- `tests/test_generate_dataset.py`: unit tests
- `data/historical_name_anchors.json`: historical given-name anchors
- `data/surname_fallback_top100.csv`: offline 2015 surname fallback
- `sample_output/synthetic_shift_dataset.xlsx`: sample workbook with five templates
- `sample_output/name_dictionary.csv`: given-name dictionary
- `sample_output/surname_dictionary.csv`: raw ranked surname dictionary
- `sample_output/surname_sampling_pool.csv`: Hangul-aggregated surname sampling pool
- `sample_output/code_dictionary.csv`: requested shift-code dictionary
- `sample_output/images/`: five Excel-rendered schedule PNGs
- `sample_output/annotations/`: row/cell JSONL and CSV ground truth
- `sample_output/test_report.txt`: unit-test report
- `sample_output/workbook_preview.png`: workbook render preview
