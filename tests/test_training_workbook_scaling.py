import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import generate_dataset as generator
import generate_training_dataset as training_generator


class TrainingWorkbookScalingTests(unittest.TestCase):
    def test_training_chunk_payload_omits_duplicated_reference_dictionaries(self):
        config = generator.GeneratorConfig(
            count=1,
            min_people=1,
            max_people=1,
            fixed_months=[2],
            ensure_all_codes=False,
            scrape_surnames=False,
        )
        names = generator.build_name_dictionary()
        surname_entries = generator.build_surname_dictionary(scrape=False)
        surname_pool = generator.aggregate_surnames(surname_entries)
        schedule = generator.generate_schedule(
            1,
            config,
            generator.random.Random(31),
            names,
            surname_pool,
            forced_template="clean_grid",
        )

        chunk = generator.build_renderer_payload(
            [schedule],
            names,
            surname_entries,
            surname_pool,
            Path("unused"),
            workbook_profile="training_chunk",
        )
        self.assertEqual(chunk["workbook_profile"], "training_chunk")
        self.assertEqual(chunk["name_entries"], [])
        self.assertEqual(chunk["surname_entries"], [])
        self.assertEqual(chunk["surname_pool"], [])

        standalone = generator.build_renderer_payload(
            [schedule], names, surname_entries, surname_pool, Path("unused")
        )
        self.assertEqual(standalone["workbook_profile"], "full")
        self.assertGreater(len(standalone["name_entries"]), 0)
        self.assertGreater(len(standalone["surname_entries"]), 0)

    def test_training_renderer_uses_scalable_chunk_profile(self):
        schedule = SimpleNamespace(
            schedule_id="schedule_0001",
            page_number=1,
            page_count=1,
            show_page_number=True,
            template_id="clean_grid",
            year=2026,
            month=1,
            day_count=31,
            rows=[object()],
            clean_image_path="images/schedule_0001_clean_grid.png",
            image_width=2400,
            image_height=1600,
        )
        config = SimpleNamespace(
            schedule_id_prefix="schedule",
            show_page_numbers=True,
            ensure_all_codes=False,
        )
        split_record = SimpleNamespace(layout_family="clean_grid", split="Train", cv_fold=0)
        split = SimpleNamespace(require=mock.Mock(return_value=split_record))
        plan = {
            "schedule_id": "schedule_0001",
            "template_id": "clean_grid",
            "schedule_index": 1,
            "seed": 31,
        }

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            training_generator, "make_schedule", return_value=schedule
        ), mock.patch.object(
            training_generator, "write_schedule_annotations"
        ), mock.patch.object(
            training_generator.generator, "render_dataset_workbook"
        ) as render:
            training_generator.render_plans(
                [plan],
                config,
                [],
                [],
                [],
                Path(temporary),
                split,
                object(),
                25,
            )

        self.assertEqual(render.call_args.kwargs["workbook_profile"], "training_chunk")

    def test_polygon_validation_error_is_streamed_to_cells_and_objects(self):
        self.assertIn("text_polygon_validation_max_error_px", training_generator.CELL_FIELDS)
        self.assertIn("text_polygon_validation_max_error_px", training_generator.OBJECT_FIELDS)

        with tempfile.TemporaryDirectory() as temporary:
            annotations = Path(temporary)
            writers = training_generator.StreamingAnnotations(annotations)
            writers.write("cells", {"schedule_id": "schedule_0001", "text_polygon_validation_max_error_px": 1.25})
            writers.write("objects", {"schedule_id": "schedule_0001", "text_polygon_validation_max_error_px": 0.5})
            writers.close()

            cell_json = json.loads((annotations / "cells.jsonl").read_text(encoding="utf-8"))
            object_json = json.loads((annotations / "objects.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(cell_json["text_polygon_validation_max_error_px"], 1.25)
            self.assertEqual(object_json["text_polygon_validation_max_error_px"], 0.5)

            with (annotations / "cells.csv").open(encoding="utf-8-sig", newline="") as handle:
                cell_csv = next(csv.DictReader(handle))
            with (annotations / "objects.csv").open(encoding="utf-8-sig", newline="") as handle:
                object_csv = next(csv.DictReader(handle))
            self.assertEqual(float(cell_csv["text_polygon_validation_max_error_px"]), 1.25)
            self.assertEqual(float(object_csv["text_polygon_validation_max_error_px"]), 0.5)


if __name__ == "__main__":
    unittest.main()
