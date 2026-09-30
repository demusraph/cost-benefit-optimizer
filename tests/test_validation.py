import unittest
from pydantic import ValidationError
from cost_benefit_optimizer.models import OptimizationRequest, CostMatrix

class TestValidationAndModels(unittest.TestCase):
    def setUp(self):
        self.valid_costs = CostMatrix(tp=100.0, fp=-50.0, tn=0.0, fn=-100.0)

    def test_valid_request_creation(self):
        req = OptimizationRequest(
            probabilities=[0.1, 0.9],
            labels=[0, 1],
            costs=self.valid_costs
        )
        self.assertEqual(len(req.probabilities), 2)
        self.assertEqual(len(req.labels), 2)

    def test_negative_probability_rejected(self):
        with self.assertRaises(ValidationError):
            OptimizationRequest(
                probabilities=[-0.01, 0.5],
                labels=[0, 1],
                costs=self.valid_costs
            )

    def test_probability_greater_than_one_rejected(self):
        with self.assertRaises(ValidationError):
            OptimizationRequest(
                probabilities=[1.0001, 0.5],
                labels=[0, 1],
                costs=self.valid_costs
            )

    def test_empty_probabilities_rejected(self):
        with self.assertRaises(ValidationError):
            OptimizationRequest(
                probabilities=[],
                labels=[],
                costs=self.valid_costs
            )

    def test_length_mismatch_probabilities_longer(self):
        with self.assertRaises(ValidationError):
            OptimizationRequest(
                probabilities=[0.1, 0.5, 0.9],
                labels=[0, 1],
                costs=self.valid_costs
            )

    def test_length_mismatch_labels_longer(self):
        with self.assertRaises(ValidationError):
            OptimizationRequest(
                probabilities=[0.1, 0.9],
                labels=[0, 1, 1],
                costs=self.valid_costs
            )

    def test_invalid_label_two(self):
        with self.assertRaises(ValidationError):
            OptimizationRequest(
                probabilities=[0.5],
                labels=[2],
                costs=self.valid_costs
            )

    def test_invalid_label_negative(self):
        with self.assertRaises(ValidationError):
            OptimizationRequest(
                probabilities=[0.5],
                labels=[-1],
                costs=self.valid_costs
            )

    def test_label_type_coercion_string_int(self):
        req = OptimizationRequest(
            probabilities=[0.5],
            labels=["1"], # Pydantic converts "1" to 1
            costs=self.valid_costs
        )
        self.assertEqual(req.labels[0], 1)

    def test_probability_float_coercion(self):
        req = OptimizationRequest(
            probabilities=["0.75"],
            labels=[1],
            costs=self.valid_costs
        )
        self.assertEqual(req.probabilities[0], 0.75)

    def test_cost_matrix_default_tn(self):
        cm = CostMatrix(tp=100.0, fp=-50.0, fn=-200.0)
        self.assertEqual(cm.tn, 0.0)

    def test_cost_matrix_as_dict(self):
        cm = CostMatrix(tp=100.0, fp=-50.0, tn=10.0, fn=-200.0)
        d = cm.as_dict()
        self.assertEqual(d["tp"], 100.0)
        self.assertEqual(d["fp"], -50.0)
        self.assertEqual(d["tn"], 10.0)
        self.assertEqual(d["fn"], -200.0)

    def test_cost_matrix_missing_tp_rejected(self):
        with self.assertRaises(ValidationError):
            CostMatrix(fp=-50.0, fn=-200.0)

    def test_cost_matrix_missing_fp_rejected(self):
        with self.assertRaises(ValidationError):
            CostMatrix(tp=100.0, fn=-200.0)

    def test_cost_matrix_missing_fn_rejected(self):
        with self.assertRaises(ValidationError):
            CostMatrix(tp=100.0, fp=-50.0)

    def test_cost_matrix_invalid_string_type(self):
        with self.assertRaises(ValidationError):
            CostMatrix(tp="abc", fp=-50.0, fn=-200.0)

    def test_request_missing_costs_rejected(self):
        with self.assertRaises(ValidationError):
            OptimizationRequest(
                probabilities=[0.5],
                labels=[1]
            )

    def test_probability_at_exact_zero_accepted(self):
        req = OptimizationRequest(probabilities=[0.0], labels=[0], costs=self.valid_costs)
        self.assertEqual(req.probabilities[0], 0.0)

    def test_probability_at_exact_one_accepted(self):
        req = OptimizationRequest(probabilities=[1.0], labels=[1], costs=self.valid_costs)
        self.assertEqual(req.probabilities[0], 1.0)

    def test_model_dump_json_serializable(self):
        req = OptimizationRequest(probabilities=[0.2, 0.8], labels=[0, 1], costs=self.valid_costs)
        dump = req.model_dump()
        self.assertIn("probabilities", dump)
        self.assertIn("costs", dump)

    def test_cost_matrix_integer_coercion_to_float(self):
        cm = CostMatrix(tp=50, fp=-10, tn=0, fn=-20)
        self.assertIsInstance(cm.tp, float)

    def test_labels_all_zero_valid(self):
        req = OptimizationRequest(probabilities=[0.1, 0.2], labels=[0, 0], costs=self.valid_costs)
        self.assertEqual(req.labels, [0, 0])

    def test_labels_all_one_valid(self):
        req = OptimizationRequest(probabilities=[0.8, 0.9], labels=[1, 1], costs=self.valid_costs)
        self.assertEqual(req.labels, [1, 1])

    def test_dict_instantiation_for_request(self):
        payload = {
            "probabilities": [0.3, 0.7],
            "labels": [0, 1],
            "costs": {"tp": 100.0, "fp": -50.0, "tn": 0.0, "fn": -100.0}
        }
        req = OptimizationRequest(**payload)
        self.assertEqual(req.costs.tp, 100.0)

    def test_non_list_probabilities_rejected(self):
        with self.assertRaises(ValidationError):
            OptimizationRequest(probabilities=0.5, labels=[1], costs=self.valid_costs)

if __name__ == '__main__':
    unittest.main()
