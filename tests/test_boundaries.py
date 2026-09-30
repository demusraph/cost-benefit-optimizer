import unittest
import numpy as np
from cost_benefit_optimizer.engine import OptimizerEngine
from cost_benefit_optimizer.models import OptimizationRequest, CostMatrix

class TestBoundaryConditions(unittest.TestCase):
    def setUp(self):
        self.standard_costs = CostMatrix(tp=100.0, fp=-50.0, tn=0.0, fn=-100.0)

    def test_single_element_positive_detected(self):
        req = OptimizationRequest(probabilities=[0.9], labels=[1], costs=self.standard_costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertLessEqual(res["optimal_threshold"], 0.9)
        self.assertEqual(res["max_emv"], 100.0)

    def test_single_element_negative_detected(self):
        req = OptimizationRequest(probabilities=[0.1], labels=[0], costs=self.standard_costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertGreater(res["optimal_threshold"], 0.1)
        self.assertEqual(res["max_emv"], 0.0)

    def test_all_positive_labels(self):
        req = OptimizationRequest(probabilities=[0.2, 0.4, 0.6, 0.8], labels=[1, 1, 1, 1], costs=self.standard_costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertEqual(res["confusion_at_optimal"]["fp"], 0)
        self.assertEqual(res["confusion_at_optimal"]["tn"], 0)
        self.assertEqual(res["optimal_threshold"], 0.0)

    def test_all_negative_labels(self):
        req = OptimizationRequest(probabilities=[0.2, 0.4, 0.6, 0.8], labels=[0, 0, 0, 0], costs=self.standard_costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertEqual(res["confusion_at_optimal"]["tp"], 0)
        self.assertEqual(res["confusion_at_optimal"]["fn"], 0)
        # To avoid any FP when all are negative, optimal threshold should be > 0.8
        self.assertGreater(res["optimal_threshold"], 0.8)

    def test_all_probabilities_zero(self):
        req = OptimizationRequest(probabilities=[0.0, 0.0, 0.0], labels=[0, 0, 1], costs=self.standard_costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertIsNotNone(res["optimal_threshold"])

    def test_all_probabilities_one(self):
        req = OptimizationRequest(probabilities=[1.0, 1.0, 1.0], labels=[0, 1, 1], costs=self.standard_costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertIsNotNone(res["optimal_threshold"])

    def test_identical_probabilities_across_classes(self):
        req = OptimizationRequest(probabilities=[0.5, 0.5, 0.5, 0.5], labels=[0, 0, 1, 1], costs=self.standard_costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertIn(res["optimal_threshold"], [0.0, 0.5, 0.51, 1.0])

    def test_negative_payoffs_only(self):
        negative_matrix = CostMatrix(tp=-10.0, fp=-50.0, tn=-5.0, fn=-100.0)
        req = OptimizationRequest(probabilities=[0.2, 0.8], labels=[0, 1], costs=negative_matrix)
        res = OptimizerEngine.run_optimization(req)
        self.assertLessEqual(res["max_emv"], 0.0)

    def test_positive_costs_for_all(self):
        positive_matrix = CostMatrix(tp=100.0, fp=10.0, tn=5.0, fn=2.0)
        req = OptimizationRequest(probabilities=[0.1, 0.9], labels=[0, 1], costs=positive_matrix)
        res = OptimizerEngine.run_optimization(req)
        self.assertGreater(res["max_emv"], 0.0)

    def test_threshold_at_exact_boundary_zero(self):
        probs = np.array([0.0, 0.5, 1.0])
        labels = np.array([0, 1, 1])
        emv = OptimizerEngine.calculate_emv(probs, labels, 0.0, self.standard_costs.as_dict())
        self.assertIsInstance(emv, float)

    def test_threshold_at_exact_boundary_one(self):
        probs = np.array([0.0, 0.5, 1.0])
        labels = np.array([0, 1, 1])
        emv = OptimizerEngine.calculate_emv(probs, labels, 1.0, self.standard_costs.as_dict())
        self.assertIsInstance(emv, float)

    def test_very_small_probabilities(self):
        req = OptimizationRequest(probabilities=[1e-5, 2e-5, 3e-5], labels=[0, 0, 1], costs=self.standard_costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertGreaterEqual(res["optimal_threshold"], 0.0)

    def test_near_one_probabilities(self):
        req = OptimizationRequest(probabilities=[0.9999, 0.9998], labels=[1, 1], costs=self.standard_costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertLessEqual(res["optimal_threshold"], 0.9999)

    def test_tn_cost_dominance(self):
        tn_heavy = CostMatrix(tp=1.0, fp=-10.0, tn=1000.0, fn=-1.0)
        req = OptimizationRequest(probabilities=[0.3, 0.7], labels=[0, 1], costs=tn_heavy)
        res = OptimizerEngine.run_optimization(req)
        self.assertGreater(res["max_emv"], 500)

    def test_tp_cost_dominance(self):
        tp_heavy = CostMatrix(tp=5000.0, fp=-1.0, tn=0.0, fn=-1.0)
        req = OptimizationRequest(probabilities=[0.1, 0.9], labels=[0, 1], costs=tp_heavy)
        res = OptimizerEngine.run_optimization(req)
        self.assertGreaterEqual(res["max_emv"], 4000)

    def test_fp_penalty_neutralization(self):
        fp_neutral = CostMatrix(tp=100.0, fp=0.0, tn=0.0, fn=-100.0)
        req = OptimizationRequest(probabilities=[0.3, 0.7], labels=[0, 1], costs=fp_neutral)
        res = OptimizerEngine.run_optimization(req)
        self.assertLessEqual(res["optimal_threshold"], 0.7)

    def test_fn_penalty_neutralization(self):
        fn_neutral = CostMatrix(tp=100.0, fp=-100.0, tn=0.0, fn=0.0)
        req = OptimizationRequest(probabilities=[0.3, 0.7], labels=[0, 1], costs=fn_neutral)
        res = OptimizerEngine.run_optimization(req)
        self.assertGreaterEqual(res["optimal_threshold"], 0.3)

    def test_symmetric_costs(self):
        sym_costs = CostMatrix(tp=50.0, fp=-50.0, tn=50.0, fn=-50.0)
        req = OptimizationRequest(probabilities=[0.2, 0.8], labels=[0, 1], costs=sym_costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertEqual(res["max_emv"], 100.0)

    def test_alternating_binary_sequence(self):
        probs = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]
        labels = [0, 1, 0, 1, 0, 1]
        req = OptimizationRequest(probabilities=probs, labels=labels, costs=self.standard_costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertEqual(res["confusion_at_optimal"]["tp"], 3)
        self.assertEqual(res["confusion_at_optimal"]["tn"], 3)

    def test_inverse_prediction_model(self):
        # Inverted model (low prob for label 1, high prob for label 0)
        probs = [0.9, 0.8, 0.1, 0.2]
        labels = [0, 0, 1, 1]
        req = OptimizationRequest(probabilities=probs, labels=labels, costs=self.standard_costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertIsNotNone(res["optimal_threshold"])

    def test_two_point_dataset(self):
        req = OptimizationRequest(probabilities=[0.4, 0.6], labels=[0, 1], costs=self.standard_costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertGreaterEqual(res["max_emv"], 0.0)

    def test_three_point_dataset(self):
        req = OptimizationRequest(probabilities=[0.2, 0.5, 0.8], labels=[0, 1, 1], costs=self.standard_costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertIn("optimal_threshold", res)

    def test_fine_step_granularity(self):
        req = OptimizationRequest(probabilities=[0.25, 0.75], labels=[0, 1], costs=self.standard_costs)
        res = OptimizerEngine.run_optimization(req, step=0.005)
        self.assertGreaterEqual(len(res["curve"]), 201)

    def test_coarse_step_granularity(self):
        req = OptimizationRequest(probabilities=[0.25, 0.75], labels=[0, 1], costs=self.standard_costs)
        res = OptimizerEngine.run_optimization(req, step=0.5)
        self.assertEqual(len(res["curve"]), 3)

    def test_identical_emv_across_thresholds(self):
        zero_matrix = CostMatrix(tp=0.0, fp=0.0, tn=0.0, fn=0.0)
        req = OptimizationRequest(probabilities=[0.1, 0.9], labels=[0, 1], costs=zero_matrix)
        res = OptimizerEngine.run_optimization(req)
        self.assertEqual(res["profit_lift_pct"], 0.0)

    def test_extreme_imbalance_1_to_100(self):
        probs = [0.05] * 99 + [0.95]
        labels = [0] * 99 + [1]
        req = OptimizationRequest(probabilities=probs, labels=labels, costs=self.standard_costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertGreater(res["optimal_threshold"], 0.05)
        self.assertEqual(res["confusion_at_optimal"]["tp"], 1)
        self.assertEqual(res["confusion_at_optimal"]["fp"], 0)

if __name__ == '__main__':
    unittest.main()
