# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.2-alpha] - 2026-05-17

This release stabilizes the OOM-RL Phase 4 pipeline following first-production validation, hardens the layered LLM analysis system, and closes a set of critical/high-priority bugs identified during the code review. It also ships a confirmed-working closed-loop feedback cycle.

14 commits, ~150 lines changed across source since v0.4.1-alpha.

### Fixed

#### OOM-RL Phase 4 — Critical & High Priority
- **Mutable default argument `evals_result=dict()`** in `LossHistoryMixin` and three PyTorch model wrappers (`pytorch_gats_plus`, `pytorch_lstm_ic_loss`, `pytorch_lstm_rank`): the shared dict caused training-loss telemetry to bleed across consecutive ensemble-member fits within a single session.
- **Synthetic target leakage**: `_execution_risk` and other internal pipeline stage names were appearing as real `ActionItem` targets in production JSON. Filtered on `target.startswith("_")` with a console warning.
- **OpenAI client proliferation**: `LLMInterface` was constructing a fresh `openai.OpenAI()` client on every API call (4 sites), creating unnecessary TLS handshakes. Replaced with a `_get_or_create_client()` pool keyed by `(api_key, base_url)`.
- **Silent unsupported action_type skips**: `FeedbackLoop._run_execute` was discarding `adjust_weights` and `trigger_search` items with only a `logger.warning`. Added `print()` output so operators can see high-confidence manual actions that require intervention.

#### TRA Model
- **`_writer` attribute loss after pickle roundtrip**: TRA model crashed on predict-only reload because `_writer` was not restored after deserialization. Restored the attribute to prevent `AttributeError` in production inference cycles.

### Added

#### Layered LLM Pipeline Enhancements
- **Connection-level timeout**: All OpenAI clients now use `httpx.Timeout(120.0, connect=10.0)` to prevent the pipeline from stalling indefinitely on unresponsive API endpoints.
- **Combo profile member filtering**: `_build_combo_profile` now loads ensemble membership from `ensemble_config.json` and injects only the diagnoses for models that actually belong to each combo, significantly reducing irrelevant context sent to the LLM.
- **`source_signals` auto-backfill**: When the Synthesizer LLM omits `source_signals` (common), the pipeline now backfills them from the structured Signal list matched by target name, restoring the traceability link for `FeedbackEvaluator`.
- **Configurable triage temperature**: LLM triage temperature is now read from `llm_config.json` as `triage_temperature` (default: `cfg["temperature"] * 0.3`), eliminating the hardcoded magic multiplier.
- **Bounds-aware tunable parameter inference**: `_NON_TUNABLE_PARAMS` hardcoded set replaced with `_load_tunable_param_names()` that derives the tunable set from `hyperparam_bounds.json`. Falls back to the static list when the file is absent.

### Changed
- **`brute_force_ensemble.py`**: Removed 5 redundant thin-wrapper functions (`split_is_oos_by_args`, `load_combo_groups`, `generate_grouped_combinations`, `run_single_backtest`, `_append_results_to_csv`) that shadowed the direct `search_utils` imports at the top of the file.

## [0.4.1-alpha] - 2026-05-05

This release introduces the OOM-RL (Out-of-Money Reinforcement Learning) closed-loop feedback system — a four-phase pipeline that converts post-trade analysis findings into executable model optimization actions, validates them in a sandboxed workspace, and promotes proven changes back to production. Alongside this, a new Multi-Agent System (MAS) for deep post-trade analysis provides the analytical backbone, with 7 specialist agents covering every facet of production data auditing.

27 commits, 101 files changed, ~22,100 lines added across source and tests since v0.4.0-alpha.

### Added

#### 1. Multi-Agent System (MAS) for Deep Post-Trade Analysis
A modular, 7-agent framework for automated production data auditing and strategic impact analysis, designed to be fully non-invasive to existing training/inference pipelines.
- 7 Specialist Agents: Market Regime, Model Health, Ensemble Evolution, Execution Quality, Portfolio Risk, Prediction Audit, and Trade Pattern analysis — each producing structured findings and alerts.
- Multi-Window Orchestration: The Coordinator handles dual-path data discovery (workspace + archive) and generates automated analysis windows (Weekly Era, 1Y, 6M, 3M, 1M).
- Cross-Agent Synthesis: A reasoning engine detects compound patterns (e.g., regime-driven alpha decay, liquidity drift) and produces prioritized, deduplicated recommendations.
- Config Ledger: Automated snapshot system tracks the evolution of hyperparameters, ensemble compositions, and strategy settings across time.
- Prediction Audit: Audits buy/sell suggestion hit rates using Qlib forward returns and model consensus analysis.
- Reporting & LLM Integration: Structured Markdown report generator with optional OpenAI integration for natural language executive summaries.

#### 2. OOM-RL Feedback System (Phases 1–4)
The complete closed-loop pipeline from data collection to automated execution:
- **Phase 1 — Data Infrastructure:** Training convergence tracking (epochs, duration, early-stop status), centralized `OperatorLog` for script execution audit trails, config diff system with impact domains and semantic labels, and OOS/Feedback Scope workspace protocols.
- **Phase 2 — Agent Signal Enhancement:** Upgraded Model Health, Ensemble Eval, Market Regime, and Prediction Audit agents with convergence detection, OOS history comparison, LOO contribution metrics, per-model hit rate analysis, and regime-switch detection.
- **Phase 3 — LLM Critic:** Signal Extractor transforms agent findings into structured inputs; LLM Critic (configurable via workspace-specific prompt/skill files) generates validated `ActionItems` with hyperparameter bounds enforcement and scope constraints.
- **Phase 4 — Execution Layer:** Playground Manager for workspace isolation, Training Adapter for safe hyperparameter application, single-model IC validation, Config Promoter for production synchronization, and Feedback Loop Orchestrator for scheduling the end-to-end cycle.

#### 3. Training Telemetry & Convergence Logging
- Robust Metric Extraction: Implemented `BestScoreCaptureHandler` to capture `best_epoch`, `best_score`, and convergence data from non-standard model architectures (ADD, custom PyTorch) that log via child loggers rather than populating `evals_result`.
- Handler Propagation: Converted from filter-based to handler-based metric capture to reliably intercept logs from sub-module loggers (e.g., `qlib.ADD`).
- Convergence Preservation: Fixed telemetry loss across predict-only cycles by preserving convergence data when no new training occurs.
- Global Training History: All training runs now append to `training_history.jsonl` for longitudinal analysis.

#### 4. Broker Analytics & Post-Trade Enhancements
- Order & Trade Log Analytics: Extended broker adapter interface with `read_orders` and `read_trades` methods. GTJA adapter supports security code formatting and prefix filtering.
- Daily Analytics Pipeline: New `prod_post_trade_analytics.py` for daily incremental processing of order and trade data into cumulative analysis logs, decoupled from settlement logic.
- Execution Timing Analysis: Granular order-to-fill latency metrics (fill rate, cancel rate, mean/median/P90 latency) integrated into the Execution Quality agent and report output.
- Closing Value Output: Final closing portfolio value now displayed in `prod_post_trade` console output for easier verification against brokerage records.

#### 5. Portfolio Analytics
- OLS Inference Statistics: `PortfolioAnalyzer` now calculates and returns t-statistics and p-values for alpha and beta in both single-factor and multi-factor regressions, displayed inline with significance stars (`***`/`**`/`*`). Reduced precision mode supported for `--shareable` reports.

#### 6. Documentation
- 5 new documentation guides (bilingual CN/EN) covering the full OOM-RL system:
  - `50_DEEP_ANALYSIS_GUIDE` — Base MAS system usage
  - `51_OOMRL_FEEDBACK_OVERVIEW` — Closed-loop feedback architecture
  - `52_OOMRL_DATA_INFRASTRUCTURE` — OperatorLog, Config Ledger, training convergence logs
  - `53_OOMRL_CRITIC_GUIDE` — Signal extraction, Critic modes, ActionItems, Skills
  - `54_OOMRL_FEEDBACK_LOOP` — Playground, Adapter, Orchestrator, Promote
- System Overview updated with OOM-RL architecture and cross-references.

### Fixed
- fix(prod): decouple order/trade analytics state from settlement config
- fix(telemetry): preserve convergence data across predict-only cycles
- fix(oomrl-phase2): validate and harden agent outputs against production data
- fix(oomrl-phase1-2): harden convergence field names, tighten guards, and fix edge cases
- fix(critic): include current hyperparam values in LLM Critic prompt
- fix: numpy 2.x compatibility for polyfit inputs in model_health and market_regime agents
- fix: refactored scripts to move `os.chdir` to `main()` to prevent import-time side effects during testing

### Testing
- ~10,000 lines of new test code across 30+ new test files, covering:
  - All 7 MAS agents with comprehensive edge case coverage
  - Full OOM-RL pipeline (signal extractor, LLM interface, action items, feedback loop, playground manager, training adapter, config promoter)
  - Fusion ledger, operator log, training history, and convergence telemetry
  - Broker analytics, post-trade processing, and execution timing
  - Environment isolation with hermetic Qlib workspace mocking

## [0.4.0-alpha] - 2026-04-14

This release introduces comprehensive hardening of our analytical modules, structural improvements to experiment lifecycle tracking, and the integration of our academic publication.

### Added
- **Interactive HTML Reports:** Added a rich, interactive Out-Of-Sample (OOS) HTML report and consensus analysis to `analyze_ensembles.py`. Evaluating complex ensemble permutations is now highly visual and intuitive.
- **`--retrain-last` Support:** Added the long-awaited `--retrain-last` flag to the rolling training pipeline. You can now selectively clear and re-train just the most recent time window without initiating a full cold start.
- **Environment Safety:** Added environment safeguarding (`env.safeguard("Rolling Train")`) to prevent accidental destructive executions in unattended workflows.
- **MTM Metrics:** Added MTM profit factor and max daily drop metrics.
- **[arXiv Integration](https://arxiv.org/abs/2604.11477):** Our research paper is now officially public. The documentation and project landing pages have been updated to include full citations.
- **Dual License Model:** Explicitly clarified the dual licensing of the repository: MIT License for source code, CC-BY-SA 4.0 License for research papers.

### Fixed
- **Resolved Data Leakage:** Fixed evaluation data leakage instances in the `rolling_train` predictions for downstream ensembles.
- **Standardized Returns:** Resolved discrepancies between arithmetic and geometric return calculations, and fixed dimension mismatch errors in Geometric IR.
- **Accurate Benchmark & Attribution:** Repaired the benchmark return chain break in `PortfolioAnalyzer` and resolved identity mapping mismatch in multi-factor attribution.
- **Execution Fidelity:** Eliminated mathematical distortions in the `execution_analyzer`, particularly around monetary slippage calculation and substitute pooling bias.
- fix: store per-mode experiment names to resolve static/rolling recorder lookup mismatch
- fix(analysis): align substitute count logic between execution analyzer and trade classifier

### Changed
- **(Breaking Change) Structured Ensemble Outputs:** Reorganized the ensemble optimization output paths into structured, per-run directories instead of a flat output space, eliminating file overwrites and ensuring better experiment provenance.
- **Configuration Standardization:** Replaced various magic numbers (e.g., risk-free rates, temporal constants) with globally centralized constants.

### Testing
- Massive expansion of analytical test suites featuring deep validation and state management mocks (now retaining 100% precision on edge cases).

> **Upgrade Note:** Existing ensemble outputs generated in prior alpha versions might need to be migrated to the new sub-directory schema. A migration mapping feature is included to assist users in transitioning existing data structures.

## [0.3.2-alpha] - 2026-03-29

This release focuses on improving the accuracy of core analysis engines, eliminating data biases, and bolstering the project's testing and architectural infrastructure.

### Added
- **Isolated Docker Test Environment:** Introduced a fully isolated Docker environment specifically for running tests via the CI pipeline. This guarantees reproduction consistency and prevents local environment data leaks.

### Fixed
- **Portfolio Analyzer Lookahead Bias:** Applied a required 1-day lag (`shift(1)`) to factor generation, eliminating T-day data leakage into T-day return explanations.
- **Risk Factor Naming & Alpha Alignment:** Renamed the `size` factor to `liquidity` to more accurately reflect trading volume data. Standardized alpha calculations to consistently use arithmetic annualization across single and multi-factor models.
- **Rolling Prediction Data Injection:** Fixed a data discrepancy issue in the rolling predict-only process to ensure all ensemble fusion predictions are consistently formatted and correctly propagated into backtests.
- **Order Generation Arithmetic:** Corrected an issue where the final "Order Generation Complete" summary reported an inaccurate total estimated expenditure. The engine now safely calculates the real total based purely on the sum of outputted buy orders.

### Changed
- **Shared Utilities Extraction:** Successfully extracted heavily duplicated logic across the ensemble workflows into centralized utility modules (`fusion_engine.py`, `backtest_report.py`, `search_utils.py`, `ensemble_utils.py`), substantially reducing the size of specific scripts.

### Testing
- Comprehensive test enhancements adapted to the refactored structure, with significant enhancement to the test boundaries to catch earlier potential mock failures.

## [0.3.1-alpha] - 2026-03-22

This release brings improvements to ensemble search strategies, unifying the training architecture, and major refactoring of analysis capabilities.

### Added
- **MinEntropy Ensemble Search:** Introduced a new MinEntropy (mRMR) based ensemble search strategy to efficiently select model combinations, alongside isolated strategy outputs for better tracking.
- **Unified Training Architecture:** Consolidated static training scripts into a single, unified entry point and harmonized the architecture for both static and rolling training records for better maintainability.
- **Advanced Ensemble Analysis:** Decoupled ensemble analysis from the core codebase and restored comprehensive visualization capabilities for better insights.

### Fixed
- **Predict-Only Mode Tracking:** Fixed an issue in static prediction-only mode; it can now successfully load the previous model from `source_record_id` and correctly save the model state for each prediction.
- **OOS Validation Integration:** Appended Out-of-Sample (OOS) validation results directly to the analysis report text files for brute force evaluations, rather than solely depending on terminal logs.
- **Excess Returns Calculation:** Corrected data loss issues by deleting duplicated excess return logic.
- **Testing & MLflow Improvements:** Stopped the unexpected root `mlruns/` directory creation during testing loops by properly patching `MLflowExpManager` in `conftest` and resolving local Qlib module mock import issues.
- **Mock Data Integrity:** Fixed the random mock data workflow for test environments to ensure data integrity during ensemble pipeline tests.

### Documentation
- **CSV Dependency Removal:** Completely removed all obsolete documentation references to CSV prediction files, shifting entirely to Qlib Recorders.
- **Script Naming Updates:** Cleaned up obsolete static training script names across the README files and updated references to the correct overarching training and predict-only scripts.

## [0.3.0-alpha] - 2026-03-17

This pre-release introduces major structural improvements designed for high flexibility and stability. The key highlights include the launch of the Rolling Training Pipeline, significant decoupling of downstream processes, a transition from localized CSV files to unified Qlib Recorders, and a massive overhaul to the automated testing infrastructure.

### Added
- **Rolling Training Logic:** Implemented robust rolling training logic with full state management to support continuous model testing over dynamic timeframes.
- **Cold Start & Model Merging:** Enabled the merging of new models for "cold start" executions and added the capability to backtest combined predictions seamlessly.
- **Demonstration Configurations:** Provided new demo rolling configuration files to set up complex rolling-train strategies effortlessly.

### Changed
- **Prediction Loading via Qlib Recorders:** Completely migrated prediction loading away from static CSV dependencies. Downstream processes now natively fetch their data directly using Qlib Recorders.
- **Downstream Script Decoupling:** Refactored the training pipeline to fully decouple downstream scripts (`order_gen.py`, `signal_ranking.py`) from specific training workflows (static vs. rolling). This provides a single, unified method for managing predictions.
- **Advanced Pre-Training Workflow:** Refactored the pre-training workflow to decouple base models. Introduced a standalone pre-training script supported by dynamic path injection and automated feature consistency validation.
- **Unified Configuration Loader:** Centralized configuration management across the platform.
- **Workspace Reorganization:** Restructured the `scripts` directory into distinct `utils` and `tools` module namespaces for better long-term maintainability.
- **Visualizations:** Translated matplotlib plotting labels natively to English and stripped out legacy redundant Chinese font settings ensuring broader user compatibility.

### Fixed
- **Order Generation Robustness:** Substantially enhanced the `order_gen.py` script for better resilience in production contexts.
- **Ensemble Data Alignment:** Eliminated prediction vs backtest inconsistencies inside ensemble scripts by integrating improved cross-sectional data alignment and robust feature normalization.
- **File Access Cleanup:** Suppressed the system's tendency to access live-trading files while executing localized backtests or ensemble fusion combinations.

### Testing
- Migrated Testing Logic from `unittest` to `pytest`, removing wide-spread deprecation warnings caused by Pydantic, Numpy, and Scipy updates.
- Deep Analytics Verifications for `portfolio_analyzer.py` and `ensemble_analyzer.py` verifying their math against hand-computed ground truths using mock signals.
- Resilient CI Mocking Architecture: Built out a safer mocking structure avoiding `sys.module` pollution in CI platforms (GitHub Actions).
- Increased structural coverage iteratively, closing major gaps in brute-force fast scripts and pre-training validations.

### Documentation
- Extended documentation defining how market settings are processed globally in analytical models.
- Added comprehensive documentation regarding the architectural merge and backtesting capabilities associated with rolling train features.

## [0.2.2-alpha] - 2026-03-08

### Added
- **Shareable Report Mode:** Introduced a new `--shareable` parameter for analysis reports. The feature redacts sensitive data (amounts, stock codes) while preserving core performance metrics, enabling safe external sharing and demos.
- **Windows Support:** Added workspace activation scripts for the Windows environment.

### Fixed
- **Trade Classification:** Fixed an issue where trade classification percentages did not sum to 100%.
- **Dependency Management:** Added missing dependencies (`seaborn`, `matplotlib`, `tqdm`) to `pyproject.toml` and `requirements.txt`, ensuring consistent execution across development environments.
- **Incremental Training CI:** Resolved `test_incremental_train` failures in GitHub Actions caused by missing Qlib data.

### Testing
- **GitHub Actions Integration:** Successfully configured automated testing workflow.
- Significantly enhanced unit tests for Brute Force, Optimizer, Order Generation, and Ensemble Fusion core modules.
- Deep testing optimization for script-level logic and utility functions.
- Moderately relaxed Codecov patch constraints to suit the current development stage.

### Documentation
- Updated project documentation and README to reflect the latest feature changes.
- Cleaned up redundant files in the root directory.

## [0.2.1-alpha] - 2026-03-04

Hotfix release for v0.2.0-alpha.

### Fixed
- **YAML Anchor Stale Reference:** Fixed `KeyError: 'strategy'` in the training pipeline caused by YAML anchor reference invalidation. The system now correctly passes configuration items via `kwargs` to `PortAnaRecord`.

### Documentation
- Improved typography, UX, and logic detail fixes for the walkthrough guide (`docs/70_WALKTHROUGH.md`), ensuring consistency between Chinese and English documents.

## [0.2.0-alpha] - 2026-03-03

> **⚠️ Deprecated:** This version has a known YAML anchor bug fixed in v0.2.1-alpha.

This is a core Alpha version containing multiple architectural refactorings, feature expansions, and extensive documentation updates. Focus areas include decoupling strategy logic from core pipelines, introducing an extensible broker adapter framework, refactoring data path configuration, and improving open-source community standards.

### Added
- **Broker Adapter Framework:** Introduced an extensible Broker Adapter framework, decoupling settlement parsing logic to support additional broker formats in the future.
- **Landing Page:** Added a standalone, beautiful project landing page.
- **Walkthrough:** New end-to-end operational guide document.
- **Community Standards:** Established open-source community norms including `CONTRIBUTING.md`, `SECURITY.md`, and Code of Conduct.

### Changed
- **Strategy Provider Architecture:** Introduced `StrategyProvider` architecture and `strategy_config.yaml`, fully decoupling backtest and order generation strategy logic from core pipelines.
- **Qlib Data Environment:** `env.init_qlib()` now supports `QLIB_DATA_DIR` and `QLIB_REGION` environment variables for configuring and decoupling Qlib's local data paths.

### Documentation
- Full documentation sync across Chinese and English for the new Strategy Provider architecture.
- Landing Page Security Headers configuration.
- Concepts & Limitations notes regarding day-frequency backtesting, Fast Brute Force's underlying limitations, and post-market broker data format requirements.

### Testing
- Added core script unit test examples and updated project dependencies (`requirements.txt`).

## [0.1.0-alpha] - 2026-02-28

Initial public release. The system architecture has been initially open-sourced, and the entire process from Qlib training to order generation and visualization has been successfully completed. It is currently in the Alpha stage, contains known bugs, and is for testing and learning purposes only.

[Unreleased]: https://github.com/DarkLink/QuantPits/compare/v0.4.1-alpha...HEAD
[0.4.1-alpha]: https://github.com/DarkLink/QuantPits/compare/v0.4.0-alpha...v0.4.1-alpha
[0.4.0-alpha]: https://github.com/DarkLink/QuantPits/compare/v0.3.2-alpha...v0.4.0-alpha
[0.3.2-alpha]: https://github.com/DarkLink/QuantPits/compare/v0.3.1-alpha...v0.3.2-alpha
[0.3.1-alpha]: https://github.com/DarkLink/QuantPits/compare/v0.3.0-alpha...v0.3.1-alpha
[0.3.0-alpha]: https://github.com/DarkLink/QuantPits/compare/v0.2.2-alpha...v0.3.0-alpha
[0.2.2-alpha]: https://github.com/DarkLink/QuantPits/compare/v0.2.1-alpha...v0.2.2-alpha
[0.2.1-alpha]: https://github.com/DarkLink/QuantPits/compare/v0.2.0-alpha...v0.2.1-alpha
[0.2.0-alpha]: https://github.com/DarkLink/QuantPits/compare/v0.1.0-alpha...v0.2.0-alpha
[0.1.0-alpha]: https://github.com/DarkLink/QuantPits/commits/v0.1.0-alpha
