import json
import tempfile
import unittest
from pathlib import Path

from shift_ocr.charset import coverage_report, load_charset, normalize_transcription
from shift_ocr.evaluation import add_confidence_intervals, choose_mobile_route, evaluate_cells
from shift_ocr.fallback_grid import align_to_headers
from shift_ocr.master_split import MasterSplit, create_master_split, write_master_split
from shift_ocr.export import candidate_profiles
from shift_ocr.models import TableStructureModel


class MasterSplitTests(unittest.TestCase):
    def make_schedules(self, count=600):
        return [
            {
                'schedule_id': f'schedule_{index:04d}',
                'template_id': f't{index % 5}',
                'layout_family': f't{index % 5}',
                'month': index % 12 + 1,
                'people_count': 18 + index % 15,
                'seed': 1000 + index,
                'capture_target': 4 if index < 100 else 3 if index < 300 else 0,
            }
            for index in range(count)
        ]

    def test_split_is_hashed_train_only_cv_and_ood_isolated(self):
        schedules = self.make_schedules()
        ood = [{
            'schedule_id': f'ood_schedule_{index:04d}', 'template_id': f'ood_{index % 4}',
            'layout_family': f'ood_{index % 4}', 'seed': index, 'capture_target': 1,
        } for index in range(20)]
        records, metadata = create_master_split(schedules, seed=7, config={'count': 600}, ood_schedules=ood)
        counts = metadata['counts']
        self.assertGreater(counts['validation'], 70)
        self.assertGreater(counts['test'], 70)
        self.assertEqual(counts['ood_layout'], 20)
        self.assertTrue(all(item.cv_fold in {0, 1, 2} for item in records if item.split == 'train'))
        self.assertTrue(all(item.cv_fold == -1 for item in records if item.split != 'train'))
        with tempfile.TemporaryDirectory() as tmp:
            write_master_split(records, metadata, Path(tmp))
            loaded = MasterSplit.load(Path(tmp) / 'master_split.jsonl')
            validation = next(item for item in records if item.split == 'validation')
            with self.assertRaises(ValueError):
                loaded.authorize(validation.schedule_id, 'train')
            with self.assertRaises(ValueError):
                loaded.authorize(validation.schedule_id, 'cv')
            loaded.authorize(validation.schedule_id, 'route')
            test = next(item for item in records if item.split == 'test')
            with self.assertRaises(ValueError):
                loaded.authorize(test.schedule_id, 'threshold')

    def test_unknown_and_declared_split_mismatch_fail(self):
        records, metadata = create_master_split(self.make_schedules(100), seed=8, config={})
        split = MasterSplit(records, metadata)
        with self.assertRaises(ValueError):
            split.require('missing')
        sample = records[0]
        wrong = 'test' if sample.split != 'test' else 'train'
        with self.assertRaises(ValueError):
            split.require(sample.schedule_id, wrong)


class CharsetAndGeometryTests(unittest.TestCase):
    def test_fixed_charset_preserves_semantic_symbols_and_nfc(self):
        charset = load_charset(Path(__file__).parents[1] / 'data' / 'korean_charset_v1.txt')
        for character in '⁺+/—-간호D12':
            self.assertIn(character, charset)
        decomposed = '가'.encode('utf-8').decode('utf-8')
        self.assertEqual(normalize_transcription(decomposed), '가')
        report = coverage_report([
            {'display_text': 'D⁺/—', 'canonical_code': 'D+'},
            {'display_text': '간호사'},
        ], charset)
        self.assertEqual(report['oov_count'], 0)
        self.assertEqual(report['code_oov_count'], 0)

    def test_glyph_mask_becomes_one_quad_not_stroke_contours(self):
        try:
            import numpy as np
            import cv2  # noqa: F401
            from shift_ocr.text_polygons import polygon_from_glyph_mask
        except ImportError as exc:
            self.skipTest(str(exc))
        mask = np.zeros((30, 60), dtype=np.uint8)
        mask[8:22, 5:12] = 255
        mask[8:22, 40:52] = 255
        polygon = polygon_from_glyph_mask(mask)
        self.assertEqual(len(polygon), 4)
        xs = [point[0] for point in polygon]
        self.assertLess(min(xs), 5)
        self.assertGreater(max(xs), 51)


class InferenceAndEvaluationTests(unittest.TestCase):
    def test_mobile_profiles_and_table_candidate_floor(self):
        profiles = candidate_profiles('recognizer')
        names = {profile.name for profile in profiles}
        self.assertIn('recognizer_dynamic_batch', names)
        self.assertIn('recognizer_w160_b16', names)
        self.assertIn('recognizer_w320_b8', names)
        self.assertIn('recognizer_w640_b4', names)
        table_profiles = candidate_profiles('table')
        self.assertTrue(any(profile.dynamic_height and profile.dynamic_width for profile in table_profiles))
        self.assertTrue(any(not profile.dynamic_height and not profile.dynamic_width for profile in table_profiles))
        with self.assertRaises(ValueError):
            TableStructureModel(top_k=1199)

    def test_dbnet_fallback_aligns_dates_and_names(self):
        items = [
            {'text': '1', 'bbox': [100, 10, 120, 30]},
            {'text': '2', 'bbox': [140, 10, 160, 30]},
            {'text': '3', 'bbox': [180, 10, 200, 30]},
            {'text': '김민지', 'bbox': [10, 50, 80, 75]},
            {'text': 'D', 'bbox': [101, 50, 120, 75]},
            {'text': 'E', 'bbox': [141, 50, 160, 75]},
            {'text': 'N', 'bbox': [181, 50, 200, 75]},
        ]
        result = align_to_headers(items)
        self.assertGreater(result['confidence'], 0)
        cells = {(item['row'], item['col']): item['text'] for item in result['cells']}
        self.assertEqual(cells[(1, 0)], '김민지')
        self.assertEqual([cells[(1, day)] for day in (1, 2, 3)], ['D', 'E', 'N'])

    def test_metrics_bootstrap_and_validation_only_route_selection(self):
        truth = [
            {'schedule_id': 's1', 'row_index': 1, 'day': 1, 'display_text': 'D'},
            {'schedule_id': 's1', 'row_index': 1, 'day': 2, 'display_text': 'E'},
            {'schedule_id': 's2', 'row_index': 1, 'day': 1, 'display_text': 'N'},
        ]
        predictions = [
            {'schedule_id': 's1', 'row_index': 1, 'col': 1, 'text': 'D'},
            {'schedule_id': 's1', 'row_index': 1, 'col': 2, 'text': 'F'},
            {'schedule_id': 's2', 'row_index': 1, 'col': 1, 'text': 'N'},
        ]
        metrics = add_confidence_intervals(evaluate_cells(truth, predictions), iterations=200)
        self.assertAlmostEqual(metrics['cell_exact_accuracy'], 2 / 3)
        self.assertEqual(metrics['confidence_intervals']['cell_exact_accuracy']['iterations'], 200)
        routes = {
            'A': {'split': 'validation', 'full_schedule_exact_accuracy': .90, 'p95_latency_ms': 600, 'peak_memory_mb': 100},
            'B': {'split': 'validation', 'full_schedule_exact_accuracy': .85, 'p95_latency_ms': 900, 'peak_memory_mb': 130},
            'C': {'split': 'validation', 'full_schedule_exact_accuracy': .904, 'p95_latency_ms': 800, 'peak_memory_mb': 110},
            'D': {'split': 'validation', 'full_schedule_exact_accuracy': .905, 'p95_latency_ms': 1000, 'peak_memory_mb': 150},
        }
        self.assertIn(choose_mobile_route(routes)['selected_route'], {'A', 'C'})
        routes['D']['split'] = 'test'
        with self.assertRaises(ValueError):
            choose_mobile_route(routes)


if __name__ == '__main__':
    unittest.main()
