import unittest
import numpy as np
from cost_benefit_optimizer.engine import OptimizerEngine, CostBenefitOptimizer
from cost_benefit_optimizer.models import OptimizationRequest, CostMatrix
from cost_benefit_optimizer.web import app
from starlette.testclient import TestClient

class TestStressAndAdvanced(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        np.random.seed(42)

    def setUp(self):
        self.costs = CostMatrix(tp=250.0, fp=-75.0, tn=5.0, fn=-500.0)

    def test_oop_wrapper_initialization_with_dict(self):
        costs_dict = {"tp": 100.0, "fp": -50.0, "tn": 0.0, "fn": -100.0}
        opt = CostBenefitOptimizer(y_true=[0, 1], y_probs=[0.2, 0.8], costs=costs_dict)
        threshold = opt.find_optimal_threshold()
        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)

    def test_oop_wrapper_initialization_with_model(self):
        opt = CostBenefitOptimizer(y_true=[0, 1], y_probs=[0.2, 0.8], costs=self.costs)
        threshold = opt.find_optimal_threshold()
        self.assertIsNotNone(threshold)

    def test_oop_wrapper_calculate_profit(self):
        opt = CostBenefitOptimizer(y_true=[0, 1], y_probs=[0.2, 0.8], costs=self.costs)
        profit = opt.calculate_profit(0.5)
        self.assertIsInstance(profit, float)

    def test_oop_wrapper_summary_report(self):
        opt = CostBenefitOptimizer(y_true=[0, 1], y_probs=[0.2, 0.8], costs=self.costs)
        report = opt.summary_report()
        self.assertIn("optimal_threshold", report)
        self.assertIn("profit_lift_pct", report)

    def test_sensitivity_analysis_structure(self):
        req = OptimizationRequest(probabilities=[0.1, 0.4, 0.7, 0.9], labels=[0, 0, 1, 1], costs=self.costs)
        sens = OptimizerEngine.sensitivity_analysis(req, variation_pct=0.25)
        self.assertIn("parameters", sens)
        self.assertIn("tp", sens["parameters"])
        self.assertIn("fp", sens["parameters"])
        self.assertIn("fn", sens["parameters"])

    def test_sensitivity_parameter_ranges(self):
        req = OptimizationRequest(probabilities=[0.1, 0.4, 0.7, 0.9], labels=[0, 0, 1, 1], costs=self.costs)
        sens = OptimizerEngine.sensitivity_analysis(req, variation_pct=0.50)
        for param in ["tp", "fp", "fn"]:
            self.assertGreaterEqual(sens["parameters"][param]["threshold_range"], 0.0)

    def test_fastapi_health_endpoint(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")

    def test_fastapi_index_html_endpoint(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("CostBenefit Optimizer", res.text)

    def test_fastapi_optimize_endpoint_valid(self):
        payload = {
            "probabilities": [0.1, 0.4, 0.8],
            "labels": [0, 0, 1],
            "costs": {"tp": 100.0, "fp": -50.0, "tn": 0.0, "fn": -100.0}
        }
        res = self.client.post("/optimize", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("optimal_threshold", data)

    def test_fastapi_optimize_endpoint_invalid(self):
        payload = {
            "probabilities": [1.5],
            "labels": [1],
            "costs": {"tp": 100.0, "fp": -50.0, "tn": 0.0, "fn": -100.0}
        }
        res = self.client.post("/optimize", json=payload)
        self.assertEqual(res.status_code, 422) # FastAPI Pydantic validation error

    def test_fastapi_sensitivity_endpoint(self):
        payload = {
            "probabilities": [0.2, 0.8],
            "labels": [0, 1],
            "costs": {"tp": 100.0, "fp": -50.0, "tn": 0.0, "fn": -100.0}
        }
        res = self.client.post("/sensitivity?variation=0.3", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["variation_pct"], 0.3)

    def test_stress_1000_samples_throughput(self):
        probs = list(np.random.uniform(0.0, 1.0, 1000))
        labels = [int(p > 0.5) for p in probs]
        req = OptimizationRequest(probabilities=probs, labels=labels, costs=self.costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertEqual(res["sample_size"], 1000)

    def test_stress_5000_samples_throughput(self):
        probs = list(np.random.uniform(0.0, 1.0, 5000))
        labels = [int(p > 0.5) for p in probs]
        req = OptimizationRequest(probabilities=probs, labels=labels, costs=self.costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertEqual(res["sample_size"], 5000)

    def test_stress_10000_samples_throughput(self):
        probs = list(np.random.uniform(0.0, 1.0, 10000))
        labels = [int(p > 0.4) for p in probs]
        req = OptimizationRequest(probabilities=probs, labels=labels, costs=self.costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertEqual(res["sample_size"], 10000)

    def test_monte_carlo_reproducibility(self):
        np.random.seed(999)
        p1 = list(np.random.uniform(0.0, 1.0, 200))
        l1 = [int(x > 0.5) for x in p1]
        req1 = OptimizationRequest(probabilities=p1, labels=l1, costs=self.costs)
        res1 = OptimizerEngine.run_optimization(req1)

        req2 = OptimizationRequest(probabilities=p1, labels=l1, costs=self.costs)
        res2 = OptimizerEngine.run_optimization(req2)
        self.assertEqual(res1["optimal_threshold"], res2["optimal_threshold"])
        self.assertEqual(res1["max_emv"], res2["max_emv"])

    def test_monotonically_increasing_emv_under_ideal_conditions(self):
        probs = [0.05, 0.95]
        labels = [0, 1]
        req = OptimizationRequest(probabilities=probs, labels=labels, costs=self.costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertGreater(res["max_emv"], 0.0)

    def test_high_noise_dataset_convergence(self):
        np.random.seed(123)
        probs = list(np.random.uniform(0.45, 0.55, 500))
        labels = list(np.random.choice([0, 1], size=500))
        req = OptimizationRequest(probabilities=probs, labels=labels, costs=self.costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertIsNotNone(res["optimal_threshold"])

    def test_numpy_array_input_to_oop_wrapper(self):
        y_true = np.array([0, 1, 0, 1])
        y_probs = np.array([0.1, 0.9, 0.2, 0.8])
        opt = CostBenefitOptimizer(y_true, y_probs, self.costs)
        self.assertEqual(opt.find_optimal_threshold(), 0.21)

    def test_zero_baseline_lift_guard(self):
        # When baseline EMV is 0.0, lift should not divide by zero
        neutral_costs = CostMatrix(tp=0.0, fp=0.0, tn=0.0, fn=0.0)
        req = OptimizationRequest(probabilities=[0.2, 0.8], labels=[0, 1], costs=neutral_costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertEqual(res["profit_lift_pct"], 0.0)

    def test_positive_profit_lift_with_negative_baseline(self):
        # Baseline at 0.5 is negative, but optimal threshold turns profit
        req = OptimizationRequest(
            probabilities=[0.1, 0.2, 0.3, 0.4],
            labels=[0, 0, 0, 0],
            costs=CostMatrix(tp=100.0, fp=-100.0, tn=0.0, fn=-100.0)
        )
        res = OptimizerEngine.run_optimization(req)
        self.assertGreaterEqual(res["max_emv"], 0.0)

    def test_curve_length_integrity_with_custom_steps(self):
        req = OptimizationRequest(probabilities=[0.2, 0.8], labels=[0, 1], costs=self.costs)
        res = OptimizerEngine.run_optimization(req, step=0.02)
        self.assertEqual(len(res["curve"]), 51)

    def test_sensitivity_zero_cost_delta_fallback(self):
        # When cost parameter is zero, delta fallback should prevent zero step
        costs = CostMatrix(tp=0.0, fp=-50.0, tn=0.0, fn=-100.0)
        req = OptimizationRequest(probabilities=[0.2, 0.8], labels=[0, 1], costs=costs)
        sens = OptimizerEngine.sensitivity_analysis(req, variation_pct=0.2)
        self.assertIn("tp", sens["parameters"])

    def test_all_probabilities_at_half(self):
        req = OptimizationRequest(probabilities=[0.5, 0.5], labels=[0, 1], costs=self.costs)
        res = OptimizerEngine.run_optimization(req)
        self.assertIn("optimal_threshold", res)

    def test_oop_wrapper_cached_result(self):
        opt = CostBenefitOptimizer([0, 1], [0.3, 0.7], self.costs)
        res1 = opt.summary_report()
        res2 = opt.summary_report()
        self.assertIs(res1, res2)

    def test_high_density_step_sweep(self):
        req = OptimizationRequest(probabilities=[0.2, 0.8], labels=[0, 1], costs=self.costs)
        res = OptimizerEngine.run_optimization(req, step=0.001)
        self.assertEqual(len(res["curve"]), 1001)

    def test_fastapi_nonexistent_route_404(self):
        res = self.client.get("/nonexistent")
        self.assertEqual(res.status_code, 404)

if __name__ == '__main__':
    unittest.main()
