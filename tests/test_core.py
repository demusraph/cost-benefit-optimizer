import unittest
import numpy as np
from cost_benefit_optimizer.engine import OptimizerEngine
from cost_benefit_optimizer.models import OptimizationRequest, CostMatrix

class TestCoreEngine(unittest.TestCase):
    def setUp(self):
        self.costs = CostMatrix(tp=100.0, fp=-50.0, tn=10.0, fn=-200.0)
        self.probs = [0.1, 0.4, 0.6, 0.8]
        self.labels = [0, 0, 1, 1]
        self.request = OptimizationRequest(
            probabilities=self.probs,
            labels=self.labels,
            costs=self.costs
        )

    def test_confusion_at_zero_threshold(self):
        probs = np.array([0.2, 0.8])
        labels = np.array([0, 1])
        conf = OptimizerEngine.calculate_confusion(probs, labels, 0.0)
        self.assertEqual(conf["tp"], 1)
        self.assertEqual(conf["fp"], 1)
        self.assertEqual(conf["tn"], 0)
        self.assertEqual(conf["fn"], 0)

    def test_confusion_at_one_threshold(self):
        probs = np.array([0.2, 0.8])
        labels = np.array([0, 1])
        conf = OptimizerEngine.calculate_confusion(probs, labels, 1.0)
        self.assertEqual(conf["tp"], 0)
        self.assertEqual(conf["fp"], 0)
        self.assertEqual(conf["tn"], 1)
        self.assertEqual(conf["fn"], 1)

    def test_confusion_intermediate_threshold(self):
        probs = np.array([0.1, 0.5, 0.9])
        labels = np.array([0, 1, 1])
        conf = OptimizerEngine.calculate_confusion(probs, labels, 0.5)
        self.assertEqual(conf["tp"], 2)
        self.assertEqual(conf["fp"], 0)
        self.assertEqual(conf["tn"], 1)
        self.assertEqual(conf["fn"], 0)

    def test_emv_manual_calculation(self):
        probs = np.array([0.7])
        labels = np.array([1])
        costs = {"tp": 50.0, "fp": -20.0, "tn": 0.0, "fn": -100.0}
        emv = OptimizerEngine.calculate_emv(probs, labels, 0.5, costs)
        self.assertEqual(emv, 50.0)

    def test_emv_false_negative_scenario(self):
        probs = np.array([0.3])
        labels = np.array([1])
        costs = {"tp": 50.0, "fp": -20.0, "tn": 0.0, "fn": -100.0}
        emv = OptimizerEngine.calculate_emv(probs, labels, 0.5, costs)
        self.assertEqual(emv, -100.0)

    def test_emv_true_negative_scenario(self):
        probs = np.array([0.2])
        labels = np.array([0])
        costs = {"tp": 50.0, "fp": -20.0, "tn": 15.0, "fn": -100.0}
        emv = OptimizerEngine.calculate_emv(probs, labels, 0.5, costs)
        self.assertEqual(emv, 15.0)

    def test_emv_false_positive_scenario(self):
        probs = np.array([0.8])
        labels = np.array([0])
        costs = {"tp": 50.0, "fp": -20.0, "tn": 15.0, "fn": -100.0}
        emv = OptimizerEngine.calculate_emv(probs, labels, 0.5, costs)
        self.assertEqual(emv, -20.0)

    def test_run_optimization_keys(self):
        res = OptimizerEngine.run_optimization(self.request)
        for key in ["optimal_threshold", "max_emv", "baseline_threshold", "baseline_emv", "profit_lift_pct", "curve"]:
            self.assertIn(key, res)

    def test_optimal_threshold_range(self):
        res = OptimizerEngine.run_optimization(self.request)
        self.assertGreaterEqual(res["optimal_threshold"], 0.0)
        self.assertLessEqual(res["optimal_threshold"], 1.0)

    def test_max_emv_greater_or_equal_to_baseline(self):
        res = OptimizerEngine.run_optimization(self.request)
        self.assertGreaterEqual(res["max_emv"], res["baseline_emv"])

    def test_sample_size_reported_correctly(self):
        res = OptimizerEngine.run_optimization(self.request)
        self.assertEqual(res["sample_size"], 4)

    def test_curve_step_resolution(self):
        res = OptimizerEngine.run_optimization(self.request, step=0.01)
        self.assertEqual(len(res["curve"]), 101)

    def test_curve_coarse_resolution(self):
        res = OptimizerEngine.run_optimization(self.request, step=0.1)
        self.assertEqual(len(res["curve"]), 11)

    def test_curve_structure(self):
        res = OptimizerEngine.run_optimization(self.request, step=0.05)
        first_point = res["curve"][0]
        self.assertIn("threshold", first_point)
        self.assertIn("emv", first_point)
        self.assertEqual(first_point["threshold"], 0.0)

    def test_perfect_separable_model(self):
        probs = [0.0, 0.0, 1.0, 1.0]
        labels = [0, 0, 1, 1]
        req = OptimizationRequest(probabilities=probs, labels=labels, costs=self.costs)
        res = OptimizerEngine.run_optimization(req)
        # Expected: 2 TP (2*100 = 200) + 2 TN (2*10 = 20) = 220
        self.assertEqual(res["max_emv"], 220.0)

    def test_optimal_threshold_separates_perfectly(self):
        probs = [0.1, 0.2, 0.8, 0.9]
        labels = [0, 0, 1, 1]
        req = OptimizationRequest(probabilities=probs, labels=labels, costs=self.costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertGreater(res["optimal_threshold"], 0.2)
        self.assertLessEqual(res["optimal_threshold"], 0.8)

    def test_high_fp_cost_drives_threshold_up(self):
        # When False Positive cost is massive, threshold should be high to avoid FP even on ambiguous negative sample
        severe_fp_cost = CostMatrix(tp=10.0, fp=-10000.0, tn=0.0, fn=-10.0)
        req = OptimizationRequest(probabilities=[0.1, 0.6, 0.7, 0.9], labels=[0, 0, 1, 1], costs=severe_fp_cost)
        res = OptimizerEngine.run_optimization(req)
        self.assertGreaterEqual(res["optimal_threshold"], 0.61)

    def test_high_fn_cost_drives_threshold_down(self):
        # When False Negative cost is massive, threshold should be low to catch all positives
        severe_fn_cost = CostMatrix(tp=10.0, fp=-1.0, tn=0.0, fn=-10000.0)
        req = OptimizationRequest(probabilities=[0.1, 0.4, 0.7, 0.9], labels=[1, 1, 1, 1], costs=severe_fn_cost)
        res = OptimizerEngine.run_optimization(req)
        self.assertLessEqual(res["optimal_threshold"], 0.1)

    def test_profit_lift_calculation(self):
        res = OptimizerEngine.run_optimization(self.request)
        self.assertIsInstance(res["profit_lift_pct"], float)

    def test_confusion_at_optimal_consistency(self):
        res = OptimizerEngine.run_optimization(self.request)
        opt_counts = res["confusion_at_optimal"]
        total_counts = opt_counts["tp"] + opt_counts["fp"] + opt_counts["tn"] + opt_counts["fn"]
        self.assertEqual(total_counts, len(self.labels))

    def test_equal_cost_matrix(self):
        flat_costs = CostMatrix(tp=1.0, fp=1.0, tn=1.0, fn=1.0)
        req = OptimizationRequest(probabilities=[0.2, 0.8], labels=[0, 1], costs=flat_costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertEqual(res["max_emv"], 2.0)

    def test_zero_costs_all(self):
        zero_costs = CostMatrix(tp=0.0, fp=0.0, tn=0.0, fn=0.0)
        req = OptimizationRequest(probabilities=[0.2, 0.8], labels=[0, 1], costs=zero_costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertEqual(res["max_emv"], 0.0)

    def test_numpy_input_conversion(self):
        probs_np = np.array([0.3, 0.7])
        labels_np = np.array([0, 1])
        req = OptimizationRequest(probabilities=probs_np.tolist(), labels=labels_np.tolist(), costs=self.costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertIn("optimal_threshold", res)

    def test_large_cost_values_precision(self):
        big_costs = CostMatrix(tp=1e7, fp=-5e6, tn=1e5, fn=-2e7)
        req = OptimizationRequest(probabilities=[0.2, 0.8], labels=[0, 1], costs=big_costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertGreater(res["max_emv"], 0)

    def test_sub_penny_cost_values(self):
        small_costs = CostMatrix(tp=0.005, fp=-0.002, tn=0.001, fn=-0.010)
        req = OptimizationRequest(probabilities=[0.2, 0.8], labels=[0, 1], costs=small_costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertIsNotNone(res["max_emv"])

    def test_baseline_matches_calculate_emv(self):
        res = OptimizerEngine.run_optimization(self.request)
        expected_baseline = OptimizerEngine.calculate_emv(
            np.array(self.probs), np.array(self.labels), 0.5, self.costs.as_dict()
        )
        self.assertAlmostEqual(res["baseline_emv"], expected_baseline, places=2)

if __name__ == '__main__':
    unittest.main()