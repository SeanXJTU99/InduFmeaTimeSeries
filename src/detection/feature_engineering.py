"""Physics-informed feature engineering for industrial time-series anomaly detection.

Transforms raw PLC process variables into physically meaningful features
that encode distillation-column health, including pressure drop, reflux
ratio, cascade temperature gradients, and flooding indices.

All tag names and parameters are fictitious.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class FeatureConfig:
    """Feature engineering parameters."""

    # Window sizes for rolling statistics (in samples)
    rolling_mean_window: int = 60
    rolling_std_window: int = 60

    # ΔP computation: which pressure tags to use (fictitious)
    top_pressure_tag: str = "PT-301"
    bottom_pressure_tag: str = "PT-302"

    # Feature groups to compute
    enabled_groups: Sequence[str] = (
        "pressure_drop",
        "reflux_ratio",
        "temperature_gradient",
        "rolling_statistics",
        "interaction_terms",
    )


class FeatureEngineer:
    """Compute physics-informed feature vectors from raw PLC streams.

    Expects a dictionary of time-aligned 1-D process variable arrays.
    Returns a feature matrix suitable for anomaly detection models.

    Usage::

        fe = FeatureEngineer()
        X = fe.transform({
            "PT-301": top_pressure,
            "PT-302": bottom_pressure,
            "FT-301": feed_flow,
            "FT-302": reflux_flow,
            "TE-301": top_temp,
            "TE-310": bottom_temp,
            "FV-301": valve_opening,
        })
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self._cfg = config or FeatureConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, pv_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert raw PV dictionary to physics-informed feature matrix.

        Args:
            pv_dict: mapping from tag name (e.g. ``'PT-301'``) to 1-D array.

        Returns:
            2-D array of shape ``(n_samples, n_features)``.
        """
        feature_list: List[np.ndarray] = []

        if "pressure_drop" in self._cfg.enabled_groups:
            dp = self._compute_delta_p(pv_dict)
            feature_list.append(dp.reshape(-1, 1))

        if "reflux_ratio" in self._cfg.enabled_groups:
            rr = self._compute_reflux_ratio(pv_dict)
            feature_list.append(rr.reshape(-1, 1))

        if "temperature_gradient" in self._cfg.enabled_groups:
            tg = self._compute_temperature_gradient(pv_dict)
            feature_list.append(tg.reshape(-1, 1))

        if "rolling_statistics" in self._cfg.enabled_groups:
            rs_feats = self._compute_rolling_stats(pv_dict)
            feature_list.append(rs_feats)

        if "interaction_terms" in self._cfg.enabled_groups:
            it_feats = self._compute_interaction_terms(pv_dict)
            feature_list.append(it_feats)

        return np.hstack(feature_list)

    def feature_names(self) -> List[str]:
        """Return human-readable names for each column in the output matrix."""
        names: List[str] = []
        if "pressure_drop" in self._cfg.enabled_groups:
            names.append("delta_p")
        if "reflux_ratio" in self._cfg.enabled_groups:
            names.append("reflux_ratio")
        if "temperature_gradient" in self._cfg.enabled_groups:
            names.append("temp_gradient")
        if "rolling_statistics" in self._cfg.enabled_groups:
            for tag in ["PT-301", "FT-301", "TE-301"]:
                names.extend([f"{tag}_rolling_mean", f"{tag}_rolling_std"])
        if "interaction_terms" in self._cfg.enabled_groups:
            names.extend(["dp_x_reflux", "dp_x_flow", "temp_x_valve"])
        return names

    # ------------------------------------------------------------------
    # Feature computation helpers
    # ------------------------------------------------------------------

    def _compute_delta_p(self, pv: Dict[str, np.ndarray]) -> np.ndarray:
        top = pv.get(self._cfg.top_pressure_tag)
        bot = pv.get(self._cfg.bottom_pressure_tag)
        if top is None or bot is None:
            return np.zeros(1)
        return bot - top

    def _compute_reflux_ratio(self, pv: Dict[str, np.ndarray]) -> np.ndarray:
        reflux = pv.get("FT-302", np.zeros(1))
        distillate = pv.get("FT-303", np.ones(1))
        distillate = np.where(np.abs(distillate) < 1e-6, 1e-6, distillate)
        return reflux / distillate

    def _compute_temperature_gradient(self, pv: Dict[str, np.ndarray]) -> np.ndarray:
        top_temp = pv.get("TE-301", np.zeros(1))
        bot_temp = pv.get("TE-310", np.zeros(1))
        return bot_temp - top_temp

    def _compute_rolling_stats(self, pv: Dict[str, np.ndarray]) -> np.ndarray:
        w = self._cfg.rolling_mean_window
        feats: List[np.ndarray] = []
        for tag in ["PT-301", "FT-301", "TE-301"]:
            arr = pv.get(tag, np.zeros(1))
            if len(arr) < w:
                mean_arr = np.full_like(arr, np.mean(arr))
                std_arr = np.full_like(arr, np.std(arr))
            else:
                mean_arr = np.array(
                    [np.mean(arr[max(0, i - w) : i + 1]) for i in range(len(arr))]
                )
                std_arr = np.array(
                    [np.std(arr[max(0, i - w) : i + 1]) for i in range(len(arr))]
                )
            feats.extend([mean_arr.reshape(-1, 1), std_arr.reshape(-1, 1)])
        return np.hstack(feats)

    def _compute_interaction_terms(self, pv: Dict[str, np.ndarray]) -> np.ndarray:
        dp = self._compute_delta_p(pv).reshape(-1, 1)
        rr = self._compute_reflux_ratio(pv).reshape(-1, 1)
        flow = pv.get("FT-301", np.ones(1)).reshape(-1, 1)
        valve = pv.get("FV-301", np.ones(1)).reshape(-1, 1)
        temp = pv.get("TE-301", np.ones(1)).reshape(-1, 1)
        return np.hstack([dp * rr, dp * flow, temp * valve])


def compute_physics_features(
    pv_dict: Dict[str, np.ndarray],
) -> np.ndarray:
    """Convenience: compute physics-informed features in one call.

    Args:
        pv_dict: mapping from tag name to 1-D array.

    Returns:
        2-D feature matrix.
    """
    return FeatureEngineer().transform(pv_dict)
