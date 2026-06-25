---
name: testing-nse500-alpha
description: Test the NSE500 Alpha Architect trading system end-to-end. Use when verifying trading signal generation, risk engine, or feature engineering changes.
---

# Testing NSE500 Alpha Architect

## Overview
This is a pure Python CLI/library (no GUI). All testing is shell-based — no browser recording needed.

## Prerequisites
- Python 3.12+
- numpy, pandas, pyarrow (core deps)
- pytest, pytest-cov, ruff (dev deps)

Install dev deps:
```bash
pip install pytest pytest-cov ruff
```

## Quick Validation

### Self-validation (13 internal checks)
```bash
cd /home/ubuntu/2026
python nse500_swing_alpha.py
```
Expected: "ALL 13 TESTS PASSED" at the end, exit code 0.

### Unit tests
```bash
python -m pytest tests/ -q --tb=short
```
Expected: 149 passed, exit code 0.

### Coverage
```bash
python -m pytest tests/ --cov=nse500_swing_alpha --cov-report=term-missing -q
```
Expected: >= 70% coverage (currently ~84%).

### Lint
```bash
ruff check .
```
Expected: "All checks passed!"

## Key Test Areas

### Risk Engine Mathematical Verification
- 1% rule: `position_size(entry_price=X, stop_loss=Y)` → `shares * (X-Y) <= capital * 0.01`
- Kelly: `fractional_kelly(win_rate=0.5, avg_win=1, avg_loss=1)` = 0.25 (min clamp)
- Circuit breaker: `portfolio_heat >= 0.08` triggers

### Regime Override Test
The crash regime MUST override ALL buy signals. Test by:
1. Creating a SignalGenerator
2. Passing strong buy predictions (ensemble_pred=0.9, ensemble_conf=0.8)
3. Setting regime dict to CRASH_RISK for all indices
4. Asserting: 0 BUY signals, all STAY_AWAY

### Edge Cases
- Empty DataFrame → FeatureEngine returns empty
- All-NaN data → Ensemble logs warning, doesn't crash
- Negative prices → DataQualityMonitor quarantines all rows
- Zero capital → system initializes without crash

## API Notes
- `RiskEngine(base_capital=300000)` — uses `base_capital` kwarg, not `capital`
- `re.position_size(entry_price=X, stop_loss=Y)` — uses `stop_loss`, not `stop_price`
- `re.check_circuit_breaker()` — takes no args, reads `self.portfolio_heat`
- `SignalGenerator()` — takes no constructor args
- `sg.generate_signals(features, regime_dict, ensemble_pred, ensemble_conf)` — regime can be dict or Series
- Ensemble `predict()` returns tuple `(predictions, confidences)` when fitted, may return tuple when unfitted

## Devin Secrets Needed
None — this is a standalone system with no external service dependencies. Uses synthetic data generation when yfinance is unavailable.

## No CI
This repo has no CI configured. All validation must be done locally via pytest and self-validation.
