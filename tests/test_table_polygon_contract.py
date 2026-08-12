import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from shift_ocr.datasets import DenseScheduleDataset
from shift_ocr.models import decode_table_candidates


class TablePolygonTargetTests(unittest.TestCase):
    def test_table_target_has_one_center_peak_and_exact_corner_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv2.imwrite(str(root / "schedule.png"), np.full((200, 400, 3), 255, np.uint8))
            polygon = [[40, 40], [360, 40], [360, 160], [40, 160]]
            record = {
                "schedule_id": "schedule_0001",
                "image_path": "schedule.png",
                "cell_polygon": polygon,
                "text_polygon": polygon,
                "row_index": 2,
                "day": 7,
                "ignore": False,
            }

            table_item = DenseScheduleDataset(
                [record], root, kind="table", training=False, long_side=400,
            )[0]
            heatmap = table_item["cell_heatmap_target"][0]
            self.assertEqual(int((heatmap == 1.0).sum()), 1)
            self.assertLess(int((heatmap > 0).sum()), 100)
            self.assertEqual(float(heatmap[25, 50]), 1.0)
            expected_offsets = torch.tensor(
                [-40, -15, 40, -15, 40, 15, -40, 15], dtype=torch.float32,
            )
            torch.testing.assert_close(table_item["corner_target"][:, 25, 50], expected_offsets)

            dbnet_item = DenseScheduleDataset(
                [record], root, kind="dbnet", training=False, long_side=400,
            )[0]
            self.assertGreater(int((dbnet_item["probability_target"] > 0.5).sum()), 1_000)

    def test_decoder_scales_corner_quad_to_requested_target_grid(self):
        heatmap = torch.full((1, 1, 4, 4), -20.0)
        heatmap[0, 0, 1, 2] = 20.0
        corners = torch.zeros((1, 8, 4, 4))
        corners[0, :, 1, 2] = torch.tensor(
            [-1.0, -0.5, 1.0, -0.5, 1.0, 0.5, -1.0, 0.5],
        )
        decoded = decode_table_candidates(
            {"cell_heatmap": heatmap, "corner_offsets": corners},
            threshold=0.2,
            top_k=1500,
            target_size=(8, 12),
        )[0]

        self.assertEqual(decoded.points.tolist(), [[1, 2]])
        torch.testing.assert_close(
            decoded.quads[0],
            torch.tensor([[3.0, 1.0], [9.0, 1.0], [9.0, 3.0], [3.0, 3.0]]),
        )


if __name__ == "__main__":
    unittest.main()
