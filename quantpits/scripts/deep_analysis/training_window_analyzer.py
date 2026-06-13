"""
Training Window Analyzer — independent rule-based analysis of data split adequacy.

Detects:
- Training window too short / too long for current market conditions
- Regime shift has occurred outside the current training window
- Validation window too short relative to training (early stopping unreliable)
- Training data staleness (anchor_date too old)
- Window parameter recommendations based on data analysis

All rules are deterministic with explicit numeric thresholds.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WindowAnalysisFinding:
    """A single finding from the TrainingWindowAnalyzer."""

    finding_type: str  # window_too_short | window_too_long | valid_window_too_short
    # anchor_stale | regime_window_mismatch | freq_incompatible
    severity: str  # critical | warning | info
    target: str  # "global" (window params are global, not per-model)
    metrics: Dict[str, Any]
    recommendation: str = ""
    context: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class TrainingWindowAnalyzer:
    """Rule-based analyzer for training window adequacy.

    Runs independently of the LLM — produces structured findings that
    SignalExtractor converts into Signals, and that the LLM Critic uses
    for context.

    Required inputs:
        - model_config.json (current window parameters)
        - training_history.jsonl (anchor_date history)
        - market regime data from MarketRegimeAgent raw_metrics
        - training_window_bounds.json (min/max constraints)
    """

    # Default thresholds (overridable via constructor kwargs)
    MIN_TRAIN_WINDOWS = 4
    MAX_TRAIN_WINDOWS = 15
    MIN_VALID_WINDOWS = 1
    MIN_VALID_TO_TRAIN_RATIO = 0.15
    MAX_ANCHOR_AGE_DAYS = 90
    WARN_ANCHOR_AGE_DAYS = 60
    HIGH_VOL_REGIME_MIN_WINDOWS = 10
    REGIME_SWITCH_MIN_WINDOWS = 8
    MAX_TRAIN_END_GAP_YEARS = 4.0       # train_end >4yr from anchor → warning
    CRITICAL_TRAIN_END_GAP_YEARS = 5.0   # train_end >5yr from anchor → critical

    def __init__(
        self,
        workspace_root: str,
        reference_date: Optional[str] = None,
    ):
        self._workspace_root = workspace_root
        self._ref_date = (
            datetime.strptime(reference_date, "%Y-%m-%d")
            if reference_date
            else datetime.now()
        )
        self._bounds = self._load_bounds()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        market_regime_metrics: Optional[Dict[str, Any]] = None,
    ) -> List[WindowAnalysisFinding]:
        """Run all window analysis rules.

        Args:
            market_regime_metrics: Raw metrics from MarketRegimeAgent.
                Should contain volatility label, regime_switches, etc.

        Returns:
            List of WindowAnalysisFinding objects.
        """
        findings: List[WindowAnalysisFinding] = []

        config = self._load_model_config()
        if not config:
            return findings

        anchor_info = self._load_anchor_info()

        findings.extend(self._check_window_bounds(config))
        findings.extend(self._check_validation_ratio(config))
        findings.extend(self._check_train_end_gap(config, market_regime_metrics))
        findings.extend(self._check_anchor_staleness(anchor_info))

        if market_regime_metrics:
            findings.extend(
                self._check_regime_window_mismatch(config, market_regime_metrics)
            )

        findings.extend(self._check_freq_compatibility(config))

        return findings

    def generate_recommendations(
        self,
        findings: List[WindowAnalysisFinding],
    ) -> Dict[str, Any]:
        """Generate actionable parameter change recommendations from findings.

        Returns a dict in ActionItem.params format:
        {"train_set_windows": {"from": 8, "to": 12}, ...}
        """
        config = self._load_model_config()
        if not config:
            return {}

        recommendations: Dict[str, Any] = {}
        current = {
            "train_set_windows": config.get("train_set_windows"),
            "valid_set_window": config.get("valid_set_window"),
            "test_set_window": config.get("test_set_window"),
        }

        bounds = self._bounds

        for finding in findings:
            if finding.finding_type == "window_too_short":
                tw = current.get("train_set_windows")
                if tw is not None and tw > 0:
                    new_val = min(int(tw * 1.5), bounds.get("train_set_windows", {}).get("max", 20))
                    if new_val != tw:
                        recommendations["train_set_windows"] = {
                            "from": tw, "to": new_val,
                        }

            elif finding.finding_type == "window_too_long":
                tw = current.get("train_set_windows")
                if tw is not None and tw > 0:
                    new_val = max(int(tw * 0.7), bounds.get("train_set_windows", {}).get("min", 2))
                    if new_val != tw:
                        recommendations["train_set_windows"] = {
                            "from": tw, "to": new_val,
                        }

            elif finding.finding_type == "valid_window_too_short":
                tw = current.get("train_set_windows", 8)
                vw = current.get("valid_set_window", 2)
                new_val = min(
                    max(vw + 1, int(tw * 0.2)),
                    bounds.get("valid_set_window", {}).get("max", 6),
                )
                if new_val != vw:
                    recommendations["valid_set_window"] = {
                        "from": vw, "to": new_val,
                    }

            elif finding.finding_type == "train_end_too_far":
                # Reduce gap by shrinking valid and/or test, NOT train.
                # Keep test >= 2.0 (required for IS/OOS search).
                vw = current.get("valid_set_window", 2)
                ts = current.get("test_set_window", 3)

                # Target: valid → 1.5, test → min 2.0
                new_valid = 1.5
                new_test = max(2.0, ts - 1.0) if ts > 2.0 else ts

                if new_valid < vw:
                    recommendations["valid_set_window"] = {
                        "from": vw, "to": new_valid,
                    }
                if new_test < ts:
                    recommendations["test_set_window"] = {
                        "from": ts, "to": new_test,
                    }

        return recommendations

    # ------------------------------------------------------------------
    # Rule: Window size bounds
    # ------------------------------------------------------------------

    def _check_window_bounds(self, config: dict) -> List[WindowAnalysisFinding]:
        findings: List[WindowAnalysisFinding] = []
        tw = config.get("train_set_windows")
        vw = config.get("valid_set_window")

        if tw is not None and tw < self.MIN_TRAIN_WINDOWS:
            severity = "critical" if tw <= 2 else "warning"
            findings.append(WindowAnalysisFinding(
                finding_type="window_too_short",
                severity=severity,
                target="global",
                metrics={
                    "current": tw,
                    "min_recommended": self.MIN_TRAIN_WINDOWS,
                },
                recommendation=(
                    f"Increase train_set_windows from {tw} to "
                    f"at least {self.MIN_TRAIN_WINDOWS}"
                ),
                context=(
                    f"Training windows ({tw}) below minimum "
                    f"({self.MIN_TRAIN_WINDOWS}). Model may not see "
                    f"enough diverse market conditions."
                ),
            ))

        if tw is not None and tw > self.MAX_TRAIN_WINDOWS:
            findings.append(WindowAnalysisFinding(
                finding_type="window_too_long",
                severity="info",
                target="global",
                metrics={
                    "current": tw,
                    "max_recommended": self.MAX_TRAIN_WINDOWS,
                },
                recommendation=(
                    f"Consider reducing train_set_windows from {tw} "
                    f"to <= {self.MAX_TRAIN_WINDOWS}"
                ),
                context=(
                    f"Training windows ({tw}) exceeds suggested max "
                    f"({self.MAX_TRAIN_WINDOWS}). Very old data may "
                    f"introduce noise from irrelevant past regimes."
                ),
            ))

        if vw is not None and vw < self.MIN_VALID_WINDOWS:
            findings.append(WindowAnalysisFinding(
                finding_type="valid_window_too_short",
                severity="warning",
                target="global",
                metrics={
                    "current": vw,
                    "min_recommended": self.MIN_VALID_WINDOWS,
                },
                recommendation=(
                    f"Increase valid_set_window from {vw} to "
                    f"at least {self.MIN_VALID_WINDOWS}"
                ),
                context=(
                    f"Validation window ({vw}) is below minimum "
                    f"({self.MIN_VALID_WINDOWS}). Early stopping "
                    f"decisions may be unreliable."
                ),
            ))

        return findings

    # ------------------------------------------------------------------
    # Rule: Validation ratio
    # ------------------------------------------------------------------

    def _check_validation_ratio(self, config: dict) -> List[WindowAnalysisFinding]:
        findings: List[WindowAnalysisFinding] = []
        tw = config.get("train_set_windows")
        vw = config.get("valid_set_window")

        if tw and vw and tw > 0:
            ratio = vw / tw
            if ratio < self.MIN_VALID_TO_TRAIN_RATIO:
                findings.append(WindowAnalysisFinding(
                    finding_type="valid_window_too_short",
                    severity="warning",
                    target="global",
                    metrics={
                        "train_windows": tw,
                        "valid_windows": vw,
                        "ratio": round(ratio, 3),
                        "min_ratio": self.MIN_VALID_TO_TRAIN_RATIO,
                    },
                    recommendation=(
                        f"Increase valid_set_window to at least "
                        f"{int(tw * self.MIN_VALID_TO_TRAIN_RATIO)}"
                    ),
                    context=(
                        f"Valid/train ratio ({ratio:.2f}) below minimum "
                        f"({self.MIN_VALID_TO_TRAIN_RATIO}). Early "
                        f"stopping may be too aggressive for the amount "
                        f"of training data."
                    ),
                ))

        return findings

    # ------------------------------------------------------------------
    # Rule: Train-end gap (slide mode)
    # ------------------------------------------------------------------

    def _check_train_end_gap(
        self,
        config: dict,
        regime_metrics: Optional[dict] = None,
    ) -> List[WindowAnalysisFinding]:
        """Check if the gap between train_end and anchor is too large.

        In slide mode: train_end = anchor - (valid + test) years.
        A large gap means the model trains on data that may be
        irrelevant to current market conditions. The fix is to
        reduce valid and/or test, NOT train.
        """
        findings: List[WindowAnalysisFinding] = []

        mode = config.get("data_slice_mode", "slide")
        if mode != "slide":
            return findings

        valid = config.get("valid_set_window", 2)
        test = config.get("test_set_window", 3)
        gap = valid + test

        # Count regime switches if available
        switch_count = 0
        if regime_metrics:
            switches = regime_metrics.get("regime_switches", {})
            if isinstance(switches, dict):
                switch_count = switches.get("switch_count", 0)
            else:
                switch_count = int(switches)

        if gap >= self.CRITICAL_TRAIN_END_GAP_YEARS:
            severity = "critical" if switch_count >= 20 else "warning"
            findings.append(WindowAnalysisFinding(
                finding_type="train_end_too_far",
                severity=severity,
                target="global",
                metrics={
                    "gap_years": gap,
                    "valid_set_window": valid,
                    "test_set_window": test,
                    "train_end_offset": f"anchor - {gap} years",
                    "regime_switches": switch_count,
                },
                recommendation=(
                    f"Training data ends {gap:.1f} years before anchor. "
                    f"Reduce valid_set_window and/or test_set_window "
                    f"to bring training data closer to present. "
                    f"Suggested: valid=1.5, test=2.0 → gap=3.5y"
                ),
                context=(
                    f"Train data ends at anchor - {gap:.1f}y. "
                    f"With {switch_count} regime switches, "
                    f"patterns from {gap:.1f} years ago may be "
                    f"irrelevant. Consider reducing valid+test "
                    f"to bring train_end forward."
                ),
            ))
        elif gap >= self.MAX_TRAIN_END_GAP_YEARS:
            severity = "info" if switch_count < 10 else "warning"
            findings.append(WindowAnalysisFinding(
                finding_type="train_end_too_far",
                severity=severity,
                target="global",
                metrics={
                    "gap_years": gap,
                    "valid_set_window": valid,
                    "test_set_window": test,
                    "regime_switches": switch_count,
                },
                recommendation=(
                    f"Training data ends {gap:.1f}y before anchor. "
                    f"Consider reducing gap to <=3.5y."
                ),
                context=(
                    f"Train data ends at anchor - {gap:.1f}y. "
                    f"Market conditions may have shifted."
                ),
            ))

        return findings

    # ------------------------------------------------------------------
    # Rule: Anchor staleness
    # ------------------------------------------------------------------

    def _check_anchor_staleness(self, anchor_info: dict) -> List[WindowAnalysisFinding]:
        findings: List[WindowAnalysisFinding] = []
        latest_anchor = anchor_info.get("latest_anchor_date")
        if not latest_anchor:
            return findings

        try:
            anchor_dt = datetime.strptime(latest_anchor, "%Y-%m-%d")
            days_ago = (self._ref_date - anchor_dt).days
        except (ValueError, TypeError):
            return findings

        if days_ago > self.MAX_ANCHOR_AGE_DAYS:
            findings.append(WindowAnalysisFinding(
                finding_type="anchor_stale",
                severity="warning",
                target="global",
                metrics={
                    "latest_anchor": latest_anchor,
                    "days_ago": days_ago,
                    "max_age_days": self.MAX_ANCHOR_AGE_DAYS,
                    "mode": anchor_info.get("data_slice_mode", "unknown"),
                },
                recommendation=(
                    "Trigger retrain to update models with fresh data. "
                    "In slide mode, retrain will auto-slide the window forward."
                ),
                context=(
                    f"Latest training anchor is {days_ago} days old "
                    f"(> {self.MAX_ANCHOR_AGE_DAYS}d threshold). "
                    f"Models may be using stale market data."
                ),
            ))
        elif days_ago > self.WARN_ANCHOR_AGE_DAYS:
            findings.append(WindowAnalysisFinding(
                finding_type="anchor_stale",
                severity="info",
                target="global",
                metrics={
                    "latest_anchor": latest_anchor,
                    "days_ago": days_ago,
                },
                recommendation="Consider scheduling a retrain soon.",
                context=(
                    f"Latest training anchor is {days_ago} days old "
                    f"(approaching {self.MAX_ANCHOR_AGE_DAYS}d threshold)."
                ),
            ))

        return findings

    # ------------------------------------------------------------------
    # Rule: Regime vs window mismatch
    # ------------------------------------------------------------------

    def _check_regime_window_mismatch(
        self,
        config: dict,
        regime_metrics: dict,
    ) -> List[WindowAnalysisFinding]:
        """Check if training window covers the current market regime well."""
        findings: List[WindowAnalysisFinding] = []

        switches = regime_metrics.get("regime_switches", {})

        # Handle both dict-form (from agent raw_metrics) and int-form
        if isinstance(switches, dict):
            switch_count = switches.get("switch_count", 0)
        else:
            switch_count = int(switches)

        vol_label = regime_metrics.get("volatility_label", "")

        train_windows = config.get("train_set_windows", 8)

        # High-vol regime needs more training data for stable estimates
        if vol_label == "High-Vol":
            if train_windows < self.HIGH_VOL_REGIME_MIN_WINDOWS:
                findings.append(WindowAnalysisFinding(
                    finding_type="regime_window_mismatch",
                    severity="warning",
                    target="global",
                    metrics={
                        "current_windows": train_windows,
                        "min_recommended": self.HIGH_VOL_REGIME_MIN_WINDOWS,
                        "volatility_label": vol_label,
                    },
                    recommendation=(
                        f"High-volatility regime — increase "
                        f"train_set_windows to at least "
                        f"{self.HIGH_VOL_REGIME_MIN_WINDOWS}"
                    ),
                    context=(
                        f"High volatility ({vol_label}) with only "
                        f"{train_windows} training windows. More data "
                        f"needed for stable parameter estimates."
                    ),
                ))

        # Frequent regime switching needs broader training coverage
        if switch_count >= 3:
            if train_windows < self.REGIME_SWITCH_MIN_WINDOWS:
                findings.append(WindowAnalysisFinding(
                    finding_type="regime_window_mismatch",
                    severity="info",
                    target="global",
                    metrics={
                        "switch_count": switch_count,
                        "current_windows": train_windows,
                        "min_recommended": self.REGIME_SWITCH_MIN_WINDOWS,
                    },
                    recommendation=(
                        f"Market has switched regimes {switch_count} times. "
                        f"Increase train_set_windows to at least "
                        f"{self.REGIME_SWITCH_MIN_WINDOWS}."
                    ),
                    context=(
                        f"Regime switched {switch_count} times. Training "
                        f"window of {train_windows} may not span enough "
                        f"market conditions for robust generalization."
                    ),
                ))

        return findings

    # ------------------------------------------------------------------
    # Rule: Frequency compatibility
    # ------------------------------------------------------------------

    def _check_freq_compatibility(self, config: dict) -> List[WindowAnalysisFinding]:
        findings: List[WindowAnalysisFinding] = []
        freq = config.get("freq", "week")

        if freq == "day" and config.get("train_set_windows", 8) > 365:
            findings.append(WindowAnalysisFinding(
                finding_type="freq_incompatible",
                severity="info",
                target="global",
                metrics={
                    "freq": freq,
                    "train_set_windows": config.get("train_set_windows"),
                },
                recommendation=(
                    "Daily frequency with large window count may cause "
                    "excessive training time. Consider weekly resampling."
                ),
                context=(
                    f"Daily freq with {config.get('train_set_windows')} "
                    f"windows may produce very large datasets."
                ),
            ))

        return findings

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_model_config(self) -> dict:
        path = os.path.join(self._workspace_root, "config", "model_config.json")
        if not os.path.exists(path):
            logger.warning("model_config.json not found at %s", path)
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load model_config.json: %s", e)
            return {}

    def _load_bounds(self) -> dict:
        path = os.path.join(
            self._workspace_root, "config", "training_window_bounds.json"
        )
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return data.get("bounds", {})
        except Exception:
            return {}

    def _load_anchor_info(self) -> dict:
        """Extract anchor_date information from training_history.jsonl."""
        history_path = os.path.join(
            self._workspace_root, "data", "training_history.jsonl"
        )
        if not os.path.exists(history_path):
            return {}

        latest_anchor = None
        anchors: List[dict] = []
        try:
            with open(history_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    anchor = rec.get("anchor_date", "")
                    if anchor:
                        anchors.append({
                            "model": rec.get("model_name", ""),
                            "anchor_date": anchor,
                            "trained_at": rec.get("trained_at", ""),
                        })
                        if not latest_anchor or anchor > latest_anchor:
                            latest_anchor = anchor
        except Exception as e:
            logger.warning("Failed to read training_history: %s", e)

        config = self._load_model_config()

        return {
            "latest_anchor_date": latest_anchor,
            "total_anchors": len(anchors),
            "samples": anchors[-5:],
            "data_slice_mode": config.get("data_slice_mode", "unknown"),
        }
