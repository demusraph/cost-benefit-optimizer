import numpy as np
from typing import Dict, Any, List, Union
from .models import OptimizationRequest, CostMatrix

class OptimizerEngine:
    """
    Stateless high-throughput vector engine for threshold sweeping and Expected Monetary Value (EMV) optimization.
    """
    @staticmethod
    def calculate_confusion(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, int]:
        """Calculates TP, FP, TN, FN counts for a given threshold."""
        preds = (probs >= threshold).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        tn = int(np.sum((preds == 0) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

    @staticmethod
    def calculate_emv(probs: np.ndarray, labels: np.ndarray, threshold: float, costs: Dict[str, float]) -> float:
        """
        Calculates Expected Monetary Value (EMV) at a decision threshold.
        EMV = (TP * Cost_TP) + (FP * Cost_FP) + (TN * Cost_TN) + (FN * Cost_FN)
        """
        counts = OptimizerEngine.calculate_confusion(probs, labels, threshold)
        return float(
            (counts['tp'] * costs['tp']) +
            (counts['fp'] * costs['fp']) +
            (counts['tn'] * costs['tn']) +
            (counts['fn'] * costs['fn'])
        )

    @classmethod
    def run_optimization(cls, request: OptimizationRequest, step: float = 0.01) -> Dict[str, Any]:
        """
        Sweeps decision thresholds from 0.00 to 1.00 and calculates the optimal threshold maximizing profit.
        """
        probs = np.array(request.probabilities, dtype=np.float64)
        labels = np.array(request.labels, dtype=np.int32)
        costs = request.costs.as_dict()

        num_steps = max(2, int(round(1.0 / step)) + 1)
        thresholds = np.linspace(0.0, 1.0, num_steps)
        emvs = [cls.calculate_emv(probs, labels, float(t), costs) for t in thresholds]

        best_idx = int(np.argmax(emvs))
        optimal_threshold = float(round(thresholds[best_idx], 4))
        max_emv = float(emvs[best_idx])

        # Baseline threshold comparison (standard 0.50)
        baseline_emv = cls.calculate_emv(probs, labels, 0.5, costs)
        emv_delta = max_emv - baseline_emv
        lift_pct = 0.0
        if abs(baseline_emv) > 1e-9:
            lift_pct = float(round((emv_delta / abs(baseline_emv)) * 100.0, 2))
        elif max_emv > baseline_emv:
            lift_pct = 100.0

        optimal_counts = cls.calculate_confusion(probs, labels, optimal_threshold)

        return {
            "optimal_threshold": optimal_threshold,
            "max_emv": round(max_emv, 2),
            "baseline_threshold": 0.5,
            "baseline_emv": round(baseline_emv, 2),
            "profit_lift_pct": lift_pct,
            "confusion_at_optimal": optimal_counts,
            "sample_size": len(labels),
            "curve": [
                {"threshold": float(round(t, 2)), "emv": float(round(v, 2))}
                for t, v in zip(thresholds, emvs)
            ]
        }

    @classmethod
    def sensitivity_analysis(cls, request: OptimizationRequest, variation_pct: float = 0.2) -> Dict[str, Any]:
        """
        Evaluates how threshold and EMV change under +-variation_pct perturbation in cost parameters.
        """
        base_res = cls.run_optimization(request)
        base_threshold = base_res["optimal_threshold"]
        base_costs = request.costs.as_dict()

        sensitivities = {}
        for param in ["tp", "fp", "fn"]:
            val = base_costs[param]
            perturbed_high = base_costs.copy()
            perturbed_low = base_costs.copy()
            delta = abs(val * variation_pct) if abs(val) > 1e-6 else 10.0

            perturbed_high[param] = val + delta
            perturbed_low[param] = val - delta

            req_high = OptimizationRequest(
                probabilities=request.probabilities,
                labels=request.labels,
                costs=CostMatrix(**perturbed_high)
            )
            req_low = OptimizationRequest(
                probabilities=request.probabilities,
                labels=request.labels,
                costs=CostMatrix(**perturbed_low)
            )

            res_high = cls.run_optimization(req_high)
            res_low = cls.run_optimization(req_low)

            sensitivities[param] = {
                "base_value": val,
                "plus_delta_threshold": res_high["optimal_threshold"],
                "minus_delta_threshold": res_low["optimal_threshold"],
                "threshold_range": round(abs(res_high["optimal_threshold"] - res_low["optimal_threshold"]), 4)
            }

        return {
            "base_optimal_threshold": base_threshold,
            "variation_pct": variation_pct,
            "parameters": sensitivities
        }

class CostBenefitOptimizer:
    """
    Object-oriented Pythonic wrapper for enterprise data science notebooks and pipelines.
    """
    def __init__(self, y_true: Union[List[int], np.ndarray], y_probs: Union[List[float], np.ndarray], costs: Union[Dict[str, float], CostMatrix]):
        if isinstance(costs, dict):
            self.cost_matrix = CostMatrix(
                tp=costs["tp"],
                fp=costs["fp"],
                tn=costs.get("tn", 0.0),
                fn=costs["fn"]
            )
        else:
            self.cost_matrix = costs

        self.request = OptimizationRequest(
            probabilities=list(np.asarray(y_probs, dtype=float)),
            labels=list(np.asarray(y_true, dtype=int)),
            costs=self.cost_matrix
        )
        self._result = None

    def find_optimal_threshold(self) -> float:
        if self._result is None:
            self._result = OptimizerEngine.run_optimization(self.request)
        return self._result["optimal_threshold"]

    def calculate_profit(self, threshold: float) -> float:
        probs = np.array(self.request.probabilities)
        labels = np.array(self.request.labels)
        return OptimizerEngine.calculate_emv(probs, labels, threshold, self.cost_matrix.as_dict())

    def summary_report(self) -> Dict[str, Any]:
        if self._result is None:
            self._result = OptimizerEngine.run_optimization(self.request)
        return self._result