import tempfile
import unittest
from pathlib import Path

import generate_dataset as gen


def fallback_surname_pool():
    return gen.aggregate_surnames(
        gen.build_surname_dictionary(scrape=False, max_rank=100, min_population=5)
    )


class ShiftDatasetGeneratorTests(unittest.TestCase):
    def test_year_range_dictionary(self):
        entries = gen.build_name_dictionary(1966, 2007, scrape_recent=False)
        years = {e.birth_year for e in entries}
        self.assertEqual(min(years), 1966)
        self.assertEqual(max(years), 2007)
        self.assertEqual(len(years), 42)
        self.assertTrue(all(e.source_method for e in entries))

    def test_surname_fallback_is_ranked(self):
        entries = gen.build_surname_dictionary(scrape=False)
        self.assertEqual(entries[0].surname, '김')
        self.assertEqual(entries[0].rank, 1)
        self.assertEqual(entries[0].population, 10689959)
        self.assertGreaterEqual(len(entries), 100)
        self.assertTrue(all(entries[i].rank <= entries[i + 1].rank for i in range(len(entries) - 1)))

    def test_parse_surname_html(self):
        html = '''
        <table>
          <tr><td>1</td><td>김(金)</td><td>10,689,959</td></tr>
          <tr><td>93</td><td>남궁(南宮)</td><td>21,308</td></tr>
        </table>
        '''
        entries = gen.parse_surname_ranking_html(html)
        self.assertEqual([(e.rank, e.surname, e.hanja, e.population) for e in entries], [
            (1, '김', '金', 10689959),
            (93, '남궁', '南宮', 21308),
        ])

    def test_surname_aggregation(self):
        entries = [
            gen.SurnameEntry(rank=1, surname='김', hanja='金', population=100),
            gen.SurnameEntry(rank=2, surname='김', hanja='钅', population=5),
        ]
        pool = gen.aggregate_surnames(entries)
        self.assertEqual(len(pool), 1)
        self.assertEqual(pool[0].population, 105)
        self.assertEqual(pool[0].best_rank, 1)
        self.assertEqual(pool[0].hanja_variants, '金|钅')

    def test_month_lengths_2026(self):
        expected = {2: 28, 4: 30, 6: 30, 9: 30, 11: 30}
        for month in range(1, 13):
            days = gen.calendar.monthrange(2026, month)[1]
            if month in expected:
                self.assertEqual(days, expected[month])
            else:
                self.assertEqual(days, 31)

    def test_case_mutation_preserves_nonletters(self):
        rng = gen.random.Random(7)
        value = gen.mutate_ascii_case('D/E12', rng, 1.0)
        self.assertEqual(value.upper(), 'D/E12')
        self.assertEqual(gen.mutate_ascii_case('연차', rng, 1.0), '연차')

    def test_unique_full_names_and_code_lengths(self):
        config = gen.GeneratorConfig(
            count=1,
            seed=9,
            output_dir='unused',
            min_people=25,
            max_people=25,
            fixed_months=[2],
            ensure_all_codes=False,
            scrape_surnames=False,
        )
        entries = gen.build_name_dictionary()
        surname_pool = fallback_surname_pool()
        schedule = gen.generate_schedule(
            1, config, gen.random.Random(9), entries, surname_pool,
            forced_template='clean_grid'
        )
        self.assertEqual(schedule.day_count, 28)
        self.assertEqual(len({r.name for r in schedule.rows}), 25)
        self.assertTrue(all(r.name.startswith(r.surname) for r in schedule.rows))
        self.assertTrue(all(r.name == r.surname + r.given_name for r in schedule.rows))
        self.assertTrue(all(r.surname_rank >= 1 for r in schedule.rows))
        self.assertTrue(all(r.surname_population >= 5 for r in schedule.rows))
        self.assertTrue(all(len(r.codes_canonical) == 28 for r in schedule.rows))
        self.assertTrue(all(len(r.codes_display) == 28 for r in schedule.rows))

    def test_all_codes_can_be_injected(self):
        config = gen.GeneratorConfig(
            count=1,
            seed=11,
            output_dir='unused',
            min_people=12,
            max_people=12,
            fixed_months=[1],
            ensure_all_codes=False,
            scrape_surnames=False,
        )
        entries = gen.build_name_dictionary()
        rng = gen.random.Random(11)
        schedule = gen.generate_schedule(
            1, config, rng, entries, fallback_surname_pool(),
            forced_template='clean_grid'
        )
        gen.ensure_code_coverage([schedule], rng, 0.5)
        observed = {code for row in schedule.rows for code in row.codes_canonical}
        self.assertTrue(set(gen.ALL_SHIFT_CODES).issubset(observed))

    def test_rare_korean_leave_codes_have_lower_random_weight(self):
        rng = gen.random.Random(29)
        samples = [gen.choose_code_from_group(rng, 'korean') for _ in range(20000)]
        rare_ratio = sum(code in gen.RARE_KOREAN_CODES for code in samples) / len(samples)
        uniform_ratio = len(gen.RARE_KOREAN_CODES) / len(gen.SHIFT_CODE_GROUPS['korean'])
        self.assertLess(rare_ratio, uniform_ratio * 0.6)
        self.assertGreater(rare_ratio, 0.05)

    def test_excel_render_creates_png_and_bboxes(self):
        try:
            gen.resolve_artifact_tool_runtime()
        except RuntimeError as exc:
            self.skipTest(str(exc))
        config = gen.GeneratorConfig(
            count=1,
            seed=13,
            output_dir='unused',
            min_people=3,
            max_people=3,
            fixed_months=[2],
            ensure_all_codes=False,
            scrape_surnames=False,
            template_ids=['parted_pdf'],
        )
        with tempfile.TemporaryDirectory() as tmp:
            config.output_dir = tmp
            schedules, _names, xlsx_path = gen.generate_dataset(config, force_template_cycle=True)
            schedule = schedules[0]
            image_path = Path(tmp) / schedule.clean_image_path
            self.assertTrue(xlsx_path.exists())
            self.assertTrue(image_path.exists())
            self.assertEqual(image_path.suffix, '.png')
            self.assertGreater(schedule.image_width, 2000)
            self.assertGreater(schedule.image_height, 300)
            self.assertEqual(len(schedule.cell_annotations), 3 * 28)
            self.assertTrue(all(len(a['bbox_px']) == 4 for a in schedule.cell_annotations))
            self.assertTrue(all(a['surname'] for a in schedule.cell_annotations))
            self.assertTrue(all(a['surname_rank'] >= 1 for a in schedule.cell_annotations))
            self.assertTrue(all(
                0 <= a['bbox_px'][0] < a['bbox_px'][2] <= schedule.image_width
                and 0 <= a['bbox_px'][1] < a['bbox_px'][3] <= schedule.image_height
                for a in schedule.cell_annotations
            ))


if __name__ == '__main__':
    unittest.main()
