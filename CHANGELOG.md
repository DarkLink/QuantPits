# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.8-alpha] - 2026-07-14

This release completes the **Phase 27 architecture baseline freeze**: post-trade settlement gains a fully deterministic state machine with recoverable transactions, and the static/CPCV training pipeline achieves crash-recoverable publication with exact receipt verification. Together with v0.4.4–v0.4.7, this closes the first-generation architecture transformation announced in v0.4.3.

16 commits, 165 files changed, ~17,600 lines added since v0.4.7-alpha.

### Added

#### Post-Trade — Deterministic Service Layer
- **Unified broker intake & command planning** (`post_trade/command.py`): Typed `PostTradeCommand` with settlement path discovery, broker adapter selection, and `--explain-plan` / `--json-plan` information routes that require zero Qlib initialization.
- **Deterministic account state transitions** (`post_trade/state.py`): CAS-protected state machine for cash, holdings, and position operations; every mutation is a pure function of (state, settlement event) with full before/after evidence.
- **Recoverable transaction & run audit** (`post_trade/transaction.py`): Atomic multi-file transaction with intent → staged writes → committed receipt → durable closure ordering. Crash at any point produces deterministic recovery.
- **Valuation provenance & as-of reconciliation** (`post_trade/valuation.py`): Closing values are attributed to exact settlement source, broker adapter version, and market-data snapshot identity.
- **Bonus share listing support**: Settlement adapter now correctly handles bonus share corporate actions without double-counting existing positions.

#### Training — Phase 27 Completion
- **Per-model record lineage & canonical identity** (`training/identity.py`): Every training record carries source workflow fingerprint, config fingerprint, and model-mode publication key traceable to exact MLflow experiment.
- **Plan-first static & CPCV command services** (`training/command.py`, `training/service.py`): `TrainingCommand` with typed plan → resolved targets → one-target runner → immutable evidence → receipt-backed publication. Information routes (`--explain-plan`, `--json-plan`, `--show-state`) are physically read-only.
- **Static & CPCV execution kernel closure**: Runner adapter accepts exactly one planned target, cannot expand scope, and cannot directly modify current records.
- **Publication & resume crash recovery**: Publication uses intent → staged postimages → committed receipt → current-record verification → durable OperatorLog closure ordering. Resume after crash at any point deterministically identifies committed vs. uncommitted state.
- **Phase 27 validation gap closure**: Source recorder integrity verification, audit initialization safety, failure terminalization, and closure-only resume path hardened against all identified fault points.

### Fixed
- fix(ensemble): enforce workspace lineage and exact combo inputs — combo records now validate source workspace identity
- fix(post-trade): account for bonus share listings in settlement reconciliation
- test: restore Python 3.8 compatibility and isolate post-trade Qlib dependencies

### Testing
- Post-trade Qlib dependencies isolated behind conditional imports for pure unit testing.
- Python 3.8 compatibility restored for new training/post-trade modules.

---

## [0.4.7-alpha] - 2026-07-10

This release decomposes the ensemble fusion monolith into a layered service architecture and introduces the workspace-safe command planning foundation that all production commands will share.

17 commits, 98 files changed, ~9,500 lines added since v0.4.6-alpha.

### Added

#### Runtime Foundation
- **`WorkspaceContext`** (`quantpits/runtime/context.py`): Explicit workspace identity with path containment, config fingerprinting, and activation lifecycle — replacing import-time `ROOT_DIR` resolution.
- **Workspace config validation contracts** (`quantpits/config/validators.py`): Normalize → validate → fingerprint pipeline for `model_config.json`, `strategy_config.yaml`, `prod_config.json`, and `ensemble_config.json`.
- **`CommandPlan` & `RunManifest`** (`quantpits/runtime/`): Typed command planning with `--explain-plan` / `--json-plan` information routes and post-execution manifest with exact identity, timing, and outcome evidence.
- **Explicit workspace activation**: `WorkspaceContext.activate()` replaces implicit `env.ROOT_DIR` binding; test bootstrap validates activation lifecycle.

#### Ensemble — Architecture Decomposition
- **Explain-plan & run manifests**: `ensemble_fusion.py --explain-plan` shows exact combo inputs, source lineage, and publication policy without initializing Qlib.
- **10-layer service extraction**:
  - `ensemble/service.py` — fusion orchestration service
  - `ensemble/pipeline.py` — single-combo pipeline
  - `ensemble/backtest.py` — backtest execution
  - `ensemble/analytics.py` — deterministic analytics helpers
  - `ensemble/reporting.py` — combo comparison, risk & leaderboard reporting
  - `ensemble/charts.py` — chart generation helpers
  - `ensemble/persistence.py` — persistence and ledger writers
  - `ensemble/runner.py` — command runner with typed failures
- **Import-time workspace side effects removed**: `ensemble_fusion.py` no longer executes `os.chdir()` or binds module-level paths at import time.

#### Order — Service Layer
- **Workspace-safe command planning**: `OrderCommand` with typed plan, deterministic opinion generation, and `--explain-plan` route.
- **Execution service & manifest lifecycle closure**: Order execution through service with atomic persistence and durable manifest.

### Fixed
- fix: restore Python 3.8–3.10 compatibility and add multi-version Docker tests (Python 3.8, 3.10, 3.12 matrix)

---

## [0.4.6-alpha] - 2026-07-09

This release delivers a complete rewrite of the Multi-Agent System (MAS) deep analysis pipeline with self-registering stages and workspace-local plugin isolation, plus foundational workspace context primitives.

8 commits, 65 files changed, ~5,900 lines added since v0.4.5-alpha.

### Added

#### MAS — 7-Stage Pipeline Rewrite
- **Self-registering stage architecture**: Stages register via decorator; discovery is automatic, eliminating hardcoded stage lists.
- **Label isolation**: Each stage operates in a labeled namespace, preventing cross-stage data contamination.
- **Training mode awareness**: Predict-only cycle detection prevents false-positive convergence alerts on weeks with no actual training.
- **Rolling training convergence tracking**: Decoupled JSONL-based per-model convergence telemetry for rolling windows.
- **Structured raw metrics**: Synthesizer uses structured `raw_metrics` instead of keyword matching; trade pattern agent uses mark-to-market PnL instead of placeholder `win_rate`.
- **Hardened checkpoint semantics**: Staged deep analysis checkpoints use atomic write-and-rename with content-hash verification.
- **Workspace-local plugins**: Plugin discovery scoped to workspace directory, preventing cross-workspace plugin contamination.

### Changed
- docs: standardize workspace configurations to `Demo_Workspace` and update all guides

---

## [0.4.5-alpha] - 2026-07-03

This release introduces the Purged Cross-Validation (CPCV) training system with concurrent-safe handler caching, and extends the rolling training pipeline with a Walk-Forward CPCV strategy.

14 commits, 51 files changed, ~14,000 lines added since v0.4.4-alpha.

### Added

#### Purged Cross-Validation Training System
- **`quantpits/scripts/cpcv_train.py`**: Complete Purged Cross-Validation training pipeline with configurable fold count, purge/embargo periods, and combinatorial fold ensemble.
- **CPCV data decoupling**: CPCV fully decoupled from legacy `data_slice_mode`; uses explicit date-range fold descriptors with CUDA memory alignment.
- **TRA label leakage prevention**: Fold ensemble now isolates label dimensions to prevent information leakage across purged boundaries.

#### Handler Cache & Concurrency
- **Centralized DataHandler cache**: Training data reuse across models sharing the same handler configuration, with workspace-scoped cache directory.
- **`HandlerProxy`**: Lazy-loading proxy for `DataHandler` instances with concurrent file locking (`fcntl`/`msvcrt`) for safe parallel training across processes.
- **Cache estimation**: Pre-flight memory estimation for handler cache based on instrument count and feature dimensions.

#### Rolling CPCV Strategy
- **Walk-Forward CPCV rolling strategy** (`rolling/strategy_cpcv.py`): Combines sliding window walk-forward with purged cross-validation per window, producing cross-validated predictions for each rolling period.

### Fixed
- fix(cpcv): resolve predict-only crash, add backtest, and fix consecutive runs
- fix(cpcv): resolve validation IC indexing errors and harden metric loss
- fix(cpcv): prevent TRA label leakage in fold ensemble

### Testing
- Unit tests for handler cache estimation and CPCV workflows.
- Coverage tests for CPCV, backtest, orchestration, and slide strategy modules.
- Graceful skip for `PurgedMTSDatasetH` when Qlib is not installed.

---

## [0.4.4-alpha] - 2026-06-21

This release overhauls model wrapper architecture with IR-based early stopping, migrates the MLflow backend to SQLite for MLflow 3.0+ compatibility, and extends deep analysis with market regime awareness and training window optimization.

16 commits, 152 files changed, ~13,900 lines added since v0.4.3-alpha.

### Added

#### Model Wrappers — IR-Based Early Stopping
- **Reorganized wrapper hierarchy** (`quantpits/model_wrappers/`): All wrappers now inherit `ICMetricMixin` with configurable IC/Rank-IC/IR loss metrics and `LossHistoryMixin` for convergence telemetry.
- **Best-params persistence**: Early-stopping wrappers save the parameters that achieved the best validation IC, not just the last epoch.
- **Multi-label YAML config preservation**: Training pipeline preserves multi-label configurations and adds TCTS IC wrapper support.
- **Ensemble group filtering**: `ensemble_fusion.py` filters models by group config before loading predictions, reducing unnecessary I/O.

#### MLflow Modernization
- **SQLite backend migration**: Default MLflow tracking backend migrated from file store to SQLite for MLflow 3.0+ compatibility. Migration utility handles existing file-store experiments.

#### Deep Analysis — Market Regime & Training Windows
- **QLIB-driven market regime analysis**: Market Regime agent now queries Qlib for benchmark returns, volatility regime classification, and regime-switch detection for training window optimization.
- **Training data split awareness**: Deep analysis pipeline detects training/validation/test split boundaries and generates window adjustment recommendations.

### Fixed
- fix(model): guard `evals_result=None` before delegating to Qlib base `fit()`
- fix(model): ensure `_writer` attr survives unpickling in `TRAModelIC` for predict-only mode
- fix(model): harden TRA/GATs wrappers against NaN — stable softmax, weight decay, and `isfinite` guards in `metric_fn`
- fix(model_wrappers): resolve training and early-stopping errors for 7 models
- fix(wrappers): save best params for IR early-stopping and handle non-string metrics
- fix(train): update pretrain and pretrain_validation segments for TabNet during sliding windows

### Testing
- Tests for `data_split_adapter`, `training_window_analyzer`, `migrate_mlflow_backend`, and `backtest_report`.
- Test coverage for `window_critic` and strengthened deep analysis tests.

## [0.4.3-alpha] - 2026-06-08

This release ships a major rolling-train architecture overhaul that eliminates session-long OOM hangs, hardens the OOM-RL Critic pipeline with diversity-aware guardrails and per-model tuning knowledge, introduces rank percentile normalization for ensemble fusion, and adds a full LLM observability stack. Python 3.8–3.10 compatibility is restored across the test suite.

25 commits, 85 files changed, ~12,600 lines added across source, tests, and docs since v0.4.2-alpha.

### Added

#### Rolling Train — Subprocess Isolation & OOM Prevention
- **Rolling subpackage extraction**: Refactored the monolithic `rolling_train.py` (1,300+ lines) into a `rolling/` subpackage (`windows`, `state`, `training`, `prediction`, `backtest`, `memory`), with a slim CLI entry point.
- **Per-window subprocess isolation**: Each training run is offloaded to a `ProcessPoolExecutor(max_workers=1)` using `spawn` context, so OS-level teardown fully reclaims memory held by Qlib `MemCache`, PyTorch CUDA cache, and Pandas DataFrames — eliminating WSL/Windows system hangs on long rolling runs.
- **Three-level memory management** (`memory.py`): per-window cleanup → per-model deep cleanup (Qlib `H.clear()` + double GC) → real-time pressure monitoring that raises `MemoryError` at 90 % system usage.
- **Model-first execution loop**: Inverted from window→model to model→window, enabling immediate prediction concatenation and cache release per model.
- **`--retrain-models`**: Rebuild specific models without affecting others, backed by `RollingState.remove_model()` for targeted state cleanup.
- **`--allow-stale-predict`**: Predict-only mode now blocks by default on untrained windows; the flag opts into best-effort prediction with old weights.
- **Window gap detection**: Concatenation no longer silently drops data when a gap exists between trained windows and current data.
- **Auto-repair truncated predictions**: During merge/daily concatenation, automatically detects windows whose predictions were truncated by the training-time anchor, reloads original weights, and re-predicts with the complete test range.

#### OOM-RL Critic Pipeline Enhancements
- **Per-model tuning knowledge** (`model_knowledge.yaml`): Records architecture family, regularization direction, known effective/ineffective parameters, and experimental history — injected into Per-Model LLM prompts so the Critic respects model-specific constraints.
- **Diversity-aware guardrails**: Orthogonal/diversifier signals (`avg_corr`, `group_label`, `is_diversifier`) from combo groups YAML and correlation matrix are injected into prompts; diversifier models (avg\_corr < 0.15) can no longer be disabled on IC alone.
- **Execution/Risk Critic**: Dedicated `execution_risk_system.md` skill and `generate_execution_risk_critique()` — execution/risk analysis now uses its own prompt instead of reusing the model critic prompt.
- **Single-variable experiment strategy**: Synthesizer and Experiment Analyzer enforce isolating each parameter change in separate Playground rounds to prevent masking (beneficial + harmful = neutral).
- **Standalone Experiment Advisor**: `LLMInterface.suggest_next_experiment()` wraps the ExperimentAnalyzer LLM for manual Playground workflows, auto-loading convergence data and persisting rounds to `experiment_history.jsonl`.
- **`model_selection` scope**: New `ModelSelectionAdapter` (disable/enable model) with LOO delta pre-check, combo linkage warnings, and config backup.
- **Deterministic combo routing override**: When OOS Calmar slope < −0.3 but Triage returns zero combos, automatically injects up to 3 combos for Per-Combo diagnosis.
- **Synthesizer output calibration**: Output volume guidance (2–5 items), blocked-actions table in reports, and `accuracy_decided` metric excluding pending verifications.

#### Ensemble Fusion
- **Rank percentile normalization**: Cross-sectional `[0, 1]` percentile rank as an alternative to Z-Score, now the default for long-only TopK strategies. NaN cells are filled with 0.5 (neutral vote), enabling union-based fusion instead of strict intersection.
- **`--norm-method` CLI**: Added to `ensemble_fusion.py`, `brute_force_ensemble.py`, and `brute_force_fast.py` (choices: `zscore` | `rank`).

#### LLM Observability
- **Trace logging system** (`llm_trace.py`, `langfuse_adapter.py`): Captures every LLM API call with duration, token counts, thinking blocks, and workspace details; unified `_call_llm` wrapper with session context tracking.
- **`--run-label`**: Added to `run_deep_analysis.py`, `run_feedback_loop.py`, `brute_force_ensemble.py`, and `brute_force_fast.py` to disambiguate same-date runs — labels flow into output filenames, trace directories, and action item history.

### Fixed
- **CUDA fork corruption** (`rolling`): Removed `torch.cuda.*` calls from parent-process cleanup that initialized CUDA, corrupting GPU detection in forked subprocesses (CatBoost fell back to CPU, conflicting with `bootstrap_type=Poisson`). Switched to `spawn` context.
- **Triage zero-combo routing**: Added `prioritized_combos` and `needs_execution_risk` to Triage JSON schema — the LLM wasn't outputting combo routes because the output format never asked for them.
- **`action_aggregator` null-params crash**: Handle null `params` from non-hyperparam ActionItems (e.g. `disable_model`, `retrain`).
- **Changelog param column**: Parse parameter name from `diff_snapshots` `key` field (format `model.param`) instead of missing `param` key.
- **Correlation matrix discovery**: Recursive walk under `output/` instead of flat glob; strip `@static` suffix from column/index names; deduplicate rows from `@static`/`@rolling` suffix collisions.
- **Python 3.8–3.10 compatibility**: Added missing `__init__.py` for `utils` and `tools` subpackages; explicit subpackage imports in `quantpits/__init__.py` for `unittest.mock.patch` resolution; replaced `mock_open` with real temp files in system-info tests.

### Changed
- **Rolling Train docs** (`30_ROLLING_TRAINING_GUIDE.md`, CN/EN): Full rewrite with quick-reference table, core concept explanation, per-mode chapters, and daily operations cheat sheet.
- **Predict-only UX**: Per-model missing-record messages now suggest `--merge` or `--retrain-models` instead of the destructive `--cold-start`.
- **`--retrain-last` scoping**: Now respects `--models` for scoped last-window retraining.

### Documentation
- 2 new documentation guides (bilingual CN/EN):
  - `55_OOMRL_WEEKLY_OPERATIONS` — Full weekly ops workflow (data import → post-trade → analysis → playground → promote → train → fusion → orders) with Playground safety model and troubleshooting FAQ.
  - `56_LLM_OBSERVABILITY_GUIDE` — LLM trace logging setup, session context, and Langfuse integration guide.
- Updated `02_BRUTE_FORCE_GUIDE`, `03_ENSEMBLE_FUSION_GUIDE` (CN/EN): normalization comparison, `--norm-method` parameter tables, and Known Limitations section.

### Testing
- **142 new model-wrapper tests**: Pure-function unit tests for `ICLoss`, `ICMetricMixin.metric_fn`, `LossHistoryMixin.fit()`, and parametrized import smoke tests for all 41 thin wrapper modules with MRO correctness checks. Gracefully skip when `torch` is absent.
- Expanded coverage for `rolling_benchmark`, `run_deep_analysis`, `feedback_loop`, `brute_force_fast`, `order_gen`, `static_train`, `run_analysis`, `search_utils`, `config_ledger`, `feedback_evaluator`, `promote_config`, `signal_extractor`, `synthesizer`, `model_selection_adapter`, and `llm_trace` (100% coverage).
- Python 3.8 CI hardening: selective `/proc/cpuinfo` mocking, `psutil` compatibility, `bytes`-decode guards.

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

[Unreleased]: https://github.com/DarkLink/QuantPits/compare/v0.4.3-alpha...HEAD
[0.4.3-alpha]: https://github.com/DarkLink/QuantPits/compare/v0.4.2-alpha...v0.4.3-alpha
[0.4.2-alpha]: https://github.com/DarkLink/QuantPits/compare/v0.4.1-alpha...v0.4.2-alpha
[0.4.1-alpha]: https://github.com/DarkLink/QuantPits/compare/v0.4.0-alpha...v0.4.1-alpha
[0.4.0-alpha]: https://github.com/DarkLink/QuantPits/compare/v0.3.2-alpha...v0.4.0-alpha
[0.3.2-alpha]: https://github.com/DarkLink/QuantPits/compare/v0.3.1-alpha...v0.3.2-alpha
[0.3.1-alpha]: https://github.com/DarkLink/QuantPits/compare/v0.3.0-alpha...v0.3.1-alpha
[0.3.0-alpha]: https://github.com/DarkLink/QuantPits/compare/v0.2.2-alpha...v0.3.0-alpha
[0.2.2-alpha]: https://github.com/DarkLink/QuantPits/compare/v0.2.1-alpha...v0.2.2-alpha
[0.2.1-alpha]: https://github.com/DarkLink/QuantPits/compare/v0.2.0-alpha...v0.2.1-alpha
[0.2.0-alpha]: https://github.com/DarkLink/QuantPits/compare/v0.1.0-alpha...v0.2.0-alpha
[0.1.0-alpha]: https://github.com/DarkLink/QuantPits/commits/v0.1.0-alpha
