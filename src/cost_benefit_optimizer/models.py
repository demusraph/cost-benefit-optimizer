from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Any, Optional

class CostMatrix(BaseModel):
    """
    Cost-Benefit payoff matrix for classification decisions.
    Values represent net revenue or loss associated with each outcome.
    """
    tp: float = Field(..., description="Payoff for True Positive (e.g. retained customer value - cost of retention offer)")
    fp: float = Field(..., description="Cost of False Positive (e.g. unnecessary marketing spend or manual review cost)")
    tn: float = Field(default=0.0, description="Payoff/Cost for True Negative (often zero)")
    fn: float = Field(..., description="Cost of False Negative (e.g. unmitigated fraud loss or churned customer LTV)")

    def as_dict(self) -> Dict[str, float]:
        return {"tp": float(self.tp), "fp": float(self.fp), "tn": float(self.tn), "fn": float(self.fn)}

class OptimizationRequest(BaseModel):
    """
    Request model containing probability scores, ground-truth binary labels,
    and the corresponding financial cost matrix.
    """
    probabilities: List[float] = Field(..., min_length=1, description="Predicted class 1 probabilities in [0.0, 1.0]")
    labels: List[int] = Field(..., min_length=1, description="Ground truth binary labels (0 or 1)")
    costs: CostMatrix

    @field_validator('probabilities')
    def check_probabilities_range(cls, v: List[float]) -> List[float]:
        for idx, p in enumerate(v):
            if p < 0.0 or p > 1.0:
                raise ValueError(f"Probability at index {idx} ({p}) must be within [0.0, 1.0]")
        return v

    @field_validator('labels')
    def check_binary_labels(cls, v: List[int]) -> List[int]:
        for idx, y in enumerate(v):
            if y not in (0, 1):
                raise ValueError(f"Label at index {idx} ({y}) must be binary 0 or 1")
        return v

    @model_validator(mode='after')
    def check_length_parity(self) -> 'OptimizationRequest':
        if len(self.probabilities) != len(self.labels):
            raise ValueError(
                f"Length mismatch: {len(self.probabilities)} probabilities provided, but {len(self.labels)} labels provided"
            )
        return self