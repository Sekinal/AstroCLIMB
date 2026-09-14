import unittest

import polars as pl

from assemble_selected import LABELS, PROBS, assemble


def probs(ids, rows):
    return pl.DataFrame({"pair_id": ids, **{c: [float(r[i]) for r in rows] for i, c in enumerate(PROBS)}})


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.sample = pl.DataFrame({"id": [4, 2, 1, 3], **{c: [99] * 4 for c in LABELS}, "Usage": ["ignored"] * 4})
        self.cx = probs([1], [[0.9, 0.1, 0, 0]])
        self.ia = probs([2, 4], [[0, 0.51, 0.49, 0], [0, 0.1, 0.1, 0.8]])
        self.ib = probs([4, 2], [[0, 0.1, 0.1, 0.8], [0, 0.05, 0.95, 0]])
        self.cc = probs([3], [[0, 0.7, 0.2, 0.1]])

    def run_assembly(self, **overrides):
        args = dict(sample=self.sample, cxi=self.cx, ixi_original=self.ia, ixi_reversed=self.ib, cxc=self.cc)
        return assemble(**(args | overrides))

    def test_reorders_by_id_and_averages_probabilities_not_hard_labels(self):
        result = self.run_assembly()
        self.assertEqual(result.columns, ["id", *LABELS])
        self.assertEqual(result["id"].to_list(), [4, 2, 1, 3])
        self.assertEqual(result.select(LABELS).rows(), [(0, 0, 0, 1), (0, 0, 1, 0), (1, 0, 0, 0), (0, 1, 0, 0)])

    def test_missing_and_extra_ids_rejected(self):
        for ids in ([4], [4, 20]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.run_assembly(ixi_reversed=probs(ids, [[0, 1, 0, 0]] * len(ids)))

    def test_duplicate_and_cross_task_ids_rejected(self):
        for replacement in (probs([3, 3], [[0, 1, 0, 0]] * 2), probs([2], [[0, 1, 0, 0]])):
            with self.assertRaises(ValueError):
                self.run_assembly(cxc=replacement)
        with self.assertRaises(ValueError):
            self.run_assembly(sample=pl.concat([self.sample, self.sample.head(1)]))

    def test_nonfinite_negative_and_unnormalized_values_rejected(self):
        for row in ([float("nan"), 0, 0, 0], [float("inf"), 0, 0, 0], [-0.1, 1.1, 0, 0], [0.2, 0.2, 0.2, 0.2]):
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.run_assembly(cxi=probs([1], [row]))

    def test_swapped_class_schema_rejected(self):
        with self.assertRaises(ValueError):
            self.run_assembly(cxi=self.cx.select("pair_id", PROBS[1], PROBS[0], *PROBS[2:]))
        with self.assertRaises(ValueError):
            self.run_assembly(sample=self.sample.select("id", LABELS[1], LABELS[0], *LABELS[2:], "Usage"))

    def test_mask_and_unexpected_label_column_rejected(self):
        with self.assertRaises(ValueError):
            self.run_assembly(cxc=probs([3], [[0.1, 0.7, 0.1, 0.1]]))
        with self.assertRaises(ValueError):
            self.run_assembly(cxi=self.cx.with_columns(pl.lit(0).alias("y")))


if __name__ == "__main__":
    unittest.main()
