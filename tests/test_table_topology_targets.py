import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from shift_ocr.datasets import DenseScheduleDataset, derive_table_topology_ids


def _record(name, polygon, *, row_index=None, day=None):
    return {
        "name": name,
        "schedule_id": "schedule_topology",
        "image_path": "schedule.png",
        "object_type": name,
        "display_text": name,
        "cell_polygon": polygon,
        "text_polygon": polygon,
        "row_index": row_index,
        "day": day,
        "ignore": False,
    }


def _box(left, top, right, bottom):
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


class GeometryTopologyTests(unittest.TestCase):
    def setUp(self):
        # The two middle header cells span both date sub-rows.  Body cells use
        # small projection variations to exercise overlap-based grouping.
        self.records = [
            _record("title", _box(0, 0, 300, 20)),
            _record("page", _box(300, 0, 340, 20)),
            _record("group_header", _box(0, 20, 40, 60)),
            _record("name_header", _box(40, 20, 100, 60)),
            _record("date_header", _box(100, 20, 140, 40), day=1),
            _record("weekday_header", _box(100, 40, 140, 60), day=1),
            _record("summary_header", _box(140, 20, 180, 60)),
            _record("group_1", _box(0, 60, 40, 80), row_index=1),
            _record("name_1", _box(40, 60.2, 100, 80.1), row_index=1),
            _record("shift_1", _box(100, 59.8, 140, 80.3), row_index=1, day=1),
            _record("summary_1", _box(140.3, 60, 180.1, 80), row_index=1),
            _record("group_2", _box(0, 80, 40, 100), row_index=2),
            _record("name_2", _box(40, 80, 100, 100), row_index=2),
            _record("shift_2", _box(100, 80, 140, 100), row_index=2, day=1),
            _record("summary_2", _box(140, 80, 180, 100), row_index=2),
        ]

    def _ids(self):
        values = derive_table_topology_ids(self.records)
        return {record["name"]: value for record, value in zip(self.records, values)}

    def test_projected_overlap_groups_rows_and_columns_without_merged_bridges(self):
        ids = self._ids()

        self.assertEqual(ids["title"][0], ids["page"][0])
        self.assertEqual(ids["group_header"][0], ids["name_header"][0])
        self.assertEqual(ids["group_header"][0], ids["summary_header"][0])
        self.assertNotEqual(ids["group_header"][0], ids["date_header"][0])
        self.assertNotEqual(ids["group_header"][0], ids["weekday_header"][0])
        self.assertNotEqual(ids["date_header"][0], ids["weekday_header"][0])

        body_row_1 = {ids[name][0] for name in ("group_1", "name_1", "shift_1", "summary_1")}
        body_row_2 = {ids[name][0] for name in ("group_2", "name_2", "shift_2", "summary_2")}
        self.assertEqual(len(body_row_1), 1)
        self.assertEqual(len(body_row_2), 1)
        self.assertNotEqual(body_row_1, body_row_2)

        for header, first, second in (
            ("group_header", "group_1", "group_2"),
            ("name_header", "name_1", "name_2"),
            ("date_header", "shift_1", "shift_2"),
            ("summary_header", "summary_1", "summary_2"),
        ):
            self.assertEqual(ids[header][1], ids[first][1])
            self.assertEqual(ids[first][1], ids[second][1])
        physical_columns = {
            ids[name][1] for name in ("group_1", "name_1", "shift_1", "summary_1")
        }
        self.assertEqual(len(physical_columns), 4)

    def test_dense_table_target_uses_geometry_not_roster_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv2.imwrite(str(root / "schedule.png"), np.full((120, 340, 3), 255, np.uint8))
            item = DenseScheduleDataset(
                self.records, root, kind="table", training=False, long_side=340,
            )[0]

        relation = item["relation_target"]
        observed = {}
        for record in self.records:
            polygon = np.asarray(record["cell_polygon"], dtype=np.float32)
            x, y = np.rint(polygon.mean(axis=0) / 4.0).astype(int)
            observed[record["name"]] = (
                int(relation[0, y, x]), int(relation[1, y, x]),
            )

        # These all had ``day=None`` in the old target and therefore collapsed
        # to column 0 despite being four different physical columns.
        self.assertEqual(len({observed[name][1] for name in (
            "group_1", "name_1", "summary_1", "page",
        )}), 4)
        # Header cells had row_index=None before.  They now retain the three
        # distinct physical header bands plus the title band.
        self.assertEqual(len({observed[name][0] for name in (
            "title", "group_header", "date_header", "weekday_header",
        )}), 4)

    def test_invalid_polygon_does_not_create_a_topology_band(self):
        records = self.records + [_record("invalid", None)]
        self.assertEqual(derive_table_topology_ids(records)[-1], (None, None))

    def test_one_degree_registered_rotation_preserves_grid_topology(self):
        baseline = derive_table_topology_ids(self.records)
        angle = np.deg2rad(1.0)
        rotation = np.asarray([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
        rotated = []
        for record in self.records:
            item = dict(record)
            item["cell_polygon"] = (
                np.asarray(record["cell_polygon"], dtype=np.float64) @ rotation.T
            ).tolist()
            rotated.append(item)
        self.assertEqual(derive_table_topology_ids(rotated), baseline)

    def test_dbnet_skips_only_explicit_low_confidence_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv2.imwrite(str(root / "schedule.png"), np.full((120, 340, 3), 255, np.uint8))
            base = _record("shift", _box(20, 20, 80, 50), row_index=1, day=1)
            base["object_type"] = "shift_code"

            cases = (
                ("absent", object(), True),
                ("null", None, True),
                ("true", True, True),
                ("false", False, False),
            )
            for name, confidence, expected_supervision in cases:
                with self.subTest(name=name):
                    record = dict(base)
                    if name != "absent":
                        record["registration_high_confidence"] = confidence
                    item = DenseScheduleDataset(
                        [record], root, kind="dbnet", training=False, long_side=340,
                    )[0]
                    has_supervision = bool(item["probability_target"].any())
                    self.assertEqual(has_supervision, expected_supervision)


if __name__ == "__main__":
    unittest.main()
