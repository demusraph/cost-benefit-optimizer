# AGENTS.md — Autonomous AI Engineering Operating Manual

This document is the authoritative, binding operational manual for autonomous AI coding agents (and human engineers) working within the `CostBenefit Optimizer` (`cboptimizer`) repository. All agent executions, code generations, refactorings, and pull requests MUST strictly adhere to the protocols, invariants, and quality gates detailed herein.

---

## 1. Project Overview & Core Mission

`CostBenefit Optimizer` (`cboptimizer`) is an enterprise-grade, deterministic cost-benefit analysis and trade-off optimization engine written in Python 3.11+. The system models complex financial, operational, and architectural trade-offs across compute, memory, infrastructure, latency, and financial metrics.

### Core Objectives
* **Algorithmic Evaluation Engine**: High-throughput computation of Net Present Value (NPV), Return on Investment (ROI), Internal Rate of Return (IRR), Marginal Cost Curves, and multi-objective Pareto frontiers.
* **Domain Schema Enforcement**: Rigorous modeling of cost structures, amortizations, inflation factors, risk weights, and non-monetary utility functions via Pydantic v2.
* **Deterministic Execution**: Zero hidden non-determinism. Seed-controlled stochastic simulations (Monte Carlo sensitivity analysis) with bit-exact reproducibility across platforms.
* **Decoupled System Architecture**: Complete separation of numerical core calculations, data persistence layers, CLI interfaces, and export pipelines.

---

## 2. Local PowerShell Commands

Agents operating in Windows PowerShell environments must use the following standard commands for setup, verification, testing, and formatting. Do not invent alternative flags or execution paths.

### 2.1 Virtual Environment & Setup
```powershell
# Create virtual environment with Python 3.11+
python -m venv .venv

# Activate environment in PowerShell
.\.venv\Scripts\Activate.ps1

# Upgrade core build tools
python -m pip install --upgrade pip setuptools wheel

# Install package in editable mode with all development dependencies
pip install -e ".[dev]"
```

### 2.2 Testing
```powershell
# Run full test suite with verbose output
pytest tests/ -vv

# Run full test suite with coverage report
pytest tests/ -vv --cov=src/cboptimizer --cov-report=term-missing --cov-report=xml

# Run a specific test file
pytest tests/unit/test_npv_calculator.py -vv

# Run a single targeted test method
pytest tests/unit/test_npv_calculator.py -k "test_discount_rate_zero_boundary" -vv
```

### 2.3 Execution & CLI
```powershell
# Execute CLI module direct import
python -m cboptimizer.cli --help

# Run optimization analysis using input TOML/JSON configuration
python -m cboptimizer.cli analyze --config .\examples\cloud_migration.toml --output .\reports\results.json

# Execute Monte Carlo sensitivity run
python -m cboptimizer.cli simulate --config .\examples\compute_node_tradeoff.toml --iterations 10000 --seed 42
```

### 2.4 Code Quality, Linting & Type Checking
```powershell
# Run Ruff for fast linting checks
ruff check src/ tests/

# Run Ruff auto-fixer for safe auto-fixable rules
ruff check --fix src/ tests/

# Check formatting compliance
ruff format --check src/ tests/

# Apply automated formatting
ruff format src/ tests/

# Run strict static type checking via Mypy
mypy src/cboptimizer
```

### 2.5 Single-Line Verification Pipeline
Run this composite PowerShell command prior to committing or finalizing any code modification:
```powershell
ruff check src/ tests/ ; if ($?) { ruff format --check src/ tests/ } ; if ($?) { mypy src/cboptimizer } ; if ($?) { pytest tests/ -vv }
```

---

## 3. Engineering Invariants

AI agents MUST obey the following non-negotiable software engineering invariants. Code that violates any invariant will be rejected automatically.

### 3.1 Zero-Placeholder Mandate
* **NO Stubs**: `pass`, `...`, `# TODO`, `# FIXME`, `# Implement later`, or empty function bodies are strictly prohibited in committed code.
* **NO Truncated Structures**: Mock data, data models, or dictionary returns must never contain `# ... remaining fields here` or truncated representations.
* **Explicit Exceptions**: If an abstract interface method must remain unimplemented in a base class, decorate it with `@abc.abstractmethod`. Banish `NotImplementedError` inside non-abstract runtime paths.

### 3.2 Typing Strictness
* **Explicit Annotations**: Every function parameter, return type, class attribute, and variable binding must be explicitly typed.
* **Modern Syntax**: Use Python 3.11+ type annotations (`list[str]`, `dict[str, float]`, `T | None`) rather than deprecated `typing.List` or `typing.Optional`.
* **Top-level Import**: Include `from __future__ import annotations` at the top of every `.py` file.
* **Strict Mypy**: `Any` is prohibited unless explicitly wrapping third-party dynamic objects, in which case it must be constrained via `cast()` and accompanied by an explanatory comment.
* **Immutable Schemas**: All data structures representing inputs or financial states must use immutable Pydantic v2 models (`frozen=True`).

Example of compliant typed function signature:
```python
from __future__ import annotations

from decimal import Decimal
from pydantic import BaseModel, ConfigDict, Field


class CashFlowPeriod(BaseModel):
    model_config = ConfigDict(frozen=True)

    period: int = Field(..., ge=0, description="Period sequence index")
    amount: Decimal = Field(..., description="Net cash flow amount in USD")


def calculate_discounted_cash_flow(
    cash_flows: list[CashFlowPeriod],
    discount_rate: Decimal,
) -> Decimal:
    """Calculates the present value of a sequence of cash flows.

    Args:
        cash_flows: List of period-bound cash flows.
        discount_rate: Annual discount rate as a decimal (e.g. 0.08 for 8%).

    Returns:
        Decimal total present value.
    """
    total_pv = Decimal("0.0")
    for flow in cash_flows:
        factor = (Decimal("1.0") + discount_rate) ** flow.period
        total_pv += flow.amount / factor
    return total_pv
```

### 3.3 File Encoding and Line Endings
* **UTF-8 Mandate**: All file reads and writes (`open()`, `Path.read_text()`, `Path.write_text()`) must explicitly specify `encoding="utf-8"`.
* **Line Ending Preservation**: Preserve existing line ending conventions (`LF` preferred across the codebase). Do not commit bulk `CRLF` conversions caused by Windows editing tools. Git attributes handle newline translation across OS platforms.

---

## 4. Self-Healing & Error Recovery Protocol

When an agent executes tests, lints, or type checks and encounters failures, it MUST execute the following deterministic recovery loop rather than attempting randomized code mutations.

```
+-------------------------------------------------------------+
|                     Failure Detected                        |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| Step 1: Capture Raw Output & Pinpoint Fault Location        |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| Step 2: Classify Failure Mode (Syntax, Type, Test, Lint)    |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| Step 3: Isolate to Target Minimal Command Run               |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| Step 4: Apply Root-Cause Code Repair                        |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| Step 5: Verify Local Target Fix                             |
+-------------------------------------------------------------+
                               |
            +------------------+------------------+
            |                                     |
    Pass    v                             Fail    v
+------------------------+              +---------------------+
| Step 6: Full Suite Gate|              | Increment Retry (<=3|
+------------------------+              +---------------------+
            |                                     |
     Pass   v                                     v
+------------------------+              Re-enter Step 2 Loop
| Complete Modification  |
+------------------------+
```

### Step-by-Step Self-Healing Workflow

1. **Capture & Parse Traceback**:
   * Inspect stdout/stderr from the executed process.
   * Identify exact file path, line number, exception type, and stack frame.
   * Do NOT guess the cause from memory; read the exact line referenced in the trace.

2. **Isolate Failure**:
   * Run only the specific failing check to speed up iteration loops:
     ```powershell
     # For Mypy failure in a single file
     mypy src/cboptimizer/engine/npv.py

     # For a single Pytest target failure
     pytest tests/unit/test_npv.py -k "test_invalid_discount_rate" -vv
     ```

3. **Diagnose According to Failure Mode**:
   * **Mypy Violation**: Check missing type annotations, incompatible variable reassignments, or missing `None` guards. Fix types explicitly without adding `# type: ignore` unless third-party library stubs are provably missing.
   * **Ruff Violation**: Read specific rule code (e.g., `F841`, `E501`, `I001`). Run `ruff check --fix` for auto-fixable rules, or manually reformat code structure for complex violations.
   * **Pytest Failure**: Check invariant assertions vs computed outputs. Trace variable values backward from the assertion failure. Verify edge conditions (zero values, negative discount rates, empty inputs).

4. **Apply Minimal Root-Cause Fix**:
   * Edit only code directly related to resolving the bug or fulfilling the requirement.
   * Avoid collateral code rewrites or unnecessary refactorings during fix cycles.

5. **Re-verify via Suite Quality Gate**:
   * Execute the full verification sequence (`ruff check`, `ruff format`, `mypy`, `pytest`).
   * If failure persists, retry max 3 times with refined diagnostic context. If still failing after 3 attempts, revert changes and log comprehensive block rationale.

---

## 5. Definition of Done (DoD) Quality Gate

A task or execution plan is considered strictly **DONE** only when every item on the following checklist evaluates to **TRUE**:

- [ ] **Functional Correctness**: All new or modified business logic matches domain specification requirements with explicit handling for edge cases (zero inputs, boundary conditions, empty collections).
- [ ] **Zero-Placeholder Guarantee**: File inspection confirms zero presence of `pass`, `...`, `TODO`, `FIXME`, or stubbed data structures.
- [ ] **Strict Typing Compliance**: `mypy src/cboptimizer` passes with zero errors under strict mode. No un-casted `Any` types introduced.
- [ ] **Linting & Formatting Passed**: `ruff check src/ tests/` and `ruff format --check src/ tests/` report zero violations.
- [ ] **100% Test Pass Rate**: All existing and newly created tests pass in `pytest tests/ -vv`.
- [ ] **Coverage Non-Regression**: Unit test coverage for `src/cboptimizer` meets or exceeds target threshold (minimum 95% statement coverage).
- [ ] **Explicit File Encoding**: All file standard I/O calls explicitly set `encoding="utf-8"`.
- [ ] **Clean Git Workspace**: No untracked cache directories (`.pytest_cache`, `.mypy_cache`, `__pycache__`), build outputs, or scratch files left in workspace.