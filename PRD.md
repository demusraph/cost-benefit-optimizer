# PRD: CostBenefit Optimizer

## 1. Executive Summary & Problem-Solution Fit
The **CostBenefit Optimizer** is a decision-science middleware designed to bridge the gap between model performance metrics (e.g., F1, AUC) and business P&L. Current ML workflows optimize for statistical parity, ignoring the asymmetric financial impact of False Positives (FP) and False Negatives (FN). This tool ingests raw probability distributions and a user-defined cost-benefit matrix to calculate the "Profit-Optimal Threshold," ensuring that every model inference maximizes net monetary gain rather than arbitrary statistical accuracy.

## 2. Target Personas & Core User Stories
*   **Persona: Data Scientist**
    *   *As a* Data Scientist, *I want to* map my model’s probability output to a dollar-value payoff curve, *so that* I can justify model deployment based on projected ROI rather than abstract AUC scores.
*   **Persona: Financial Analyst**
    *   *As a* Financial Analyst, *I want to* define the cost of a False Positive (e.g., customer churn incentive cost) and a False Negative (e.g., lost lifetime value), *so that* the model aligns with our quarterly budget constraints.
*   **Persona: Product Manager**
    *   *As a* Product Manager, *I want to* perform "What-If" sensitivity analysis on threshold shifts, *so that* I can understand the risk-reward trade-off of being more aggressive or conservative in our automated decisions.

## 3. MoSCoW Feature Matrix

| Priority | Feature | Description |
| :--- | :--- | :--- |
| **P0** | **Payoff Engine** | Core logic to calculate $Profit = (TP \times R_{tp}) - (FP \times C_{fp}) - (FN \times C_{fn})$. |
| **P0** | **Threshold Sweeper** | Iterative evaluation of thresholds [0.00, 1.00] at 0.01 granularity. |
| **P0** | **Matrix Validation** | Schema enforcement for cost-benefit inputs to prevent negative-profit logic errors. |
| **P1** | **Sensitivity Slider** | Interactive UI/CLI component to simulate threshold drift impact on total profit. |
| **P1** | **Multi-Class Support** | Extension of payoff logic to N-class confusion matrices. |
| **P2** | **Executive Report** | Automated PDF/JSON export summarizing "Current vs. Optimal" financial impact. |
| **P2** | **Auto-Calibration** | Gradient-based optimization to find the global maximum profit threshold without brute-force sweeping. |

## 4. Non-Functional Requirements
*   **Latency (p95):** Threshold optimization for a dataset of 1M rows must complete in < 200ms on standard hardware (4 vCPU, 8GB RAM).
*   **Memory Footprint:** Must support streaming evaluation for datasets exceeding RAM capacity using generator-based processing.
*   **Offline-Readiness:** The core library must function in air-gapped environments with zero external API dependencies.
*   **Type Safety:** 100% Python type-hinted (PEP 484) with `mypy` strict mode enabled.
*   **Modularity:** Decoupled architecture separating the `Calculator` engine from the `Visualization` and `Reporting` layers.

## 5. Quantitative Success Metrics & KPIs
*   **Model Alignment Accuracy:** Delta between predicted ROI and actual realized ROI in production (Target: < 5% variance).
*   **Decision Velocity:** Time taken for a Data Scientist to move from model training to business-aligned threshold deployment (Target: < 15 minutes).
*   **Profit Lift:** Percentage increase in net profit compared to the default 0.5 threshold (Target: > 3% improvement in baseline scenarios).
*   **Computational Efficiency:** Throughput of threshold evaluation (Target: > 500k rows/second).