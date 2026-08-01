"""Regression tests for the Apple proxy model and chart inputs."""

import math
import os
import unittest
import xml.etree.ElementTree as etree

from bench import apple_project
from bench import apple_svg


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AppleModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = apple_project.build_rows()
        cls.by_chip = {row["chip"]: row for row in cls.rows}

    def test_anchor_uses_heun_cell_evaluations(self):
        expected = 2048 * 2048 * 100 * 2 / 7.99095925
        self.assertAlmostEqual(apple_project.ANCHOR_THROUGHPUT, expected, places=7)
        anchor = self.by_chip["M4"]
        self.assertEqual(anchor["status"], "measured")
        self.assertEqual(anchor["value"], anchor["low"])
        self.assertEqual(anchor["value"], anchor["high"])

    def test_public_bandwidth_inputs_keep_provenance_precision(self):
        self.assertEqual(apple_project.BANDWIDTH["M1"], 67.0)
        self.assertEqual(apple_project.BANDWIDTH["M3 Ultra"], 819.2)
        self.assertEqual(apple_project.BANDWIDTH["M4"], 120.0)

    def test_correlated_quantizations_are_collapsed(self):
        expected = math.sqrt(13.54 * 24.11)
        self.assertAlmostEqual(apple_project.q_family("M4", "tg"), expected)
        self.assertAlmostEqual(apple_project.M5_MLX_GENERATION, 1.01 / 0.89)

    def test_key_model_outputs(self):
        # Rounded M cell-evals/s values independently audited against the model
        # definition. A change here must be accompanied by input provenance.
        expected = {
            "M1": (61.3, 55.5, 84.0),
            "M3 Ultra 60c": (521.1, 435.5, 716.6),
            "M3 Ultra": (731.5, 445.9, 839.8),
            "M4 Pro 16c": (208.8, 168.0, 238.8),
            "M4 Pro": (235.3, 209.8, 260.3),
            "M4 Max 32c": (333.5, 321.9, 358.7),
            "M4 Max 40c": (413.9, 389.3, 477.6),
            "M5": (123.2, 105.0, 133.8),
            "M5 Pro 16c": (237.9, 168.0, 269.6),
            "M5 Pro": (252.0, 210.0, 273.2),
            "M5 Max 32c": (383.3, 335.9, 402.4),
            "M5 Max 40c": (475.9, 419.9, 537.1),
        }
        for chip, triplet in expected.items():
            row = self.by_chip[chip]
            actual = tuple(row[key] / 1e6 for key in ("value", "low", "high"))
            for observed, target in zip(actual, triplet):
                self.assertAlmostEqual(observed, target, delta=0.051, msg=chip)

    def test_envelope_contains_all_coherent_proxy_paths(self):
        for chip, row in self.by_chip.items():
            if chip == "M4":
                continue
            generation, tier = apple_project.proxy_ratios(chip)
            low_ratio = row["low"] / apple_project.ANCHOR_THROUGHPUT
            high_ratio = row["high"] / apple_project.ANCHOR_THROUGHPUT
            for family in generation.keys() & tier.keys():
                path = generation[family] * tier[family]
                self.assertLessEqual(low_ratio - 1e-12, path, (chip, family))
                self.assertGreaterEqual(high_ratio + 1e-12, path, (chip, family))

    def test_price_chart_resolves_exact_launch_configs(self):
        apple_rows = apple_svg.read_apple(os.path.join(ROOT, "bench", "apple.txt"))
        tiers, order = apple_svg.read_tiers(
            os.path.join(ROOT, "bench", "price_tiers.txt"), apple_rows
        )
        self.assertEqual(order, ["entry", "mid", "high", "top"])
        apple = [row for row in tiers if row["vendor"] == "apple"]
        self.assertEqual([row["total"] for row in apple], [599, 1399, 1999, 3999])
        self.assertEqual(
            [round(row["value"], 1) for row in apple],
            [105.0, 208.8, 333.5, 521.1],
        )

    def test_measured_m4_is_not_duplicated_in_combined_chart(self):
        gpu_path = os.path.join(ROOT, "bench", "gpus.txt")
        without_apple = apple_svg.read_gpus(gpu_path)
        with_apple = apple_svg.read_gpus(gpu_path, include_apple=True)
        self.assertFalse(any("MacBook Air M4" in row["name"] for row in without_apple))
        measured_m4 = [
            row for row in with_apple if "MacBook Air M4" in row["name"]
        ]
        self.assertEqual(len(measured_m4), 1)
        self.assertEqual(measured_m4[0]["status"], "measured")

    def test_generated_svgs_are_well_formed(self):
        paths = [
            "bench/apple.svg",
            "bench/apple-vs-gpus.svg",
            "bench/apple-price-tiers.svg",
            "bench/apple-capacity.svg",
            "doc/static/gpus.svg",
        ]
        for relative in paths:
            etree.parse(os.path.join(ROOT, relative))

        with open(
            os.path.join(ROOT, "bench", "apple-capacity.svg"), encoding="utf-8"
        ) as handle:
            capacity_svg = handle.read()
        self.assertIn("702.2 M cells", capacity_svg)
        self.assertNotIn("702.2 M cell-evals/s", capacity_svg)


if __name__ == "__main__":
    unittest.main()
