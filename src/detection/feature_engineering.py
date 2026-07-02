"""Physics-informed feature engineering for industrial time-series anomaly detection.

Transforms raw PLC process variables into physically meaningful features
that encode distillation-column health, including pressure drop, reflux
ratio, cascade temperature gradients, and flooding indices.

All tag names and parameters are fictitious.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Sequence


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

        Raises:
            ValueError: if *pv_dict* is empty or no feature groups are enabled.
        """
        if not pv_dict:
            raise ValueError("pv_dict must not be empty")
        # Infer the expected signal length from whatever tags are present.
        n = self._infer_length(pv_dict)
        feature_list: List[np.ndarray] = []
        dp: np.ndarray | None = None
        rr: np.ndarray | None = None

        if "pressure_drop" in self._cfg.enabled_groups:
            dp = self._compute_delta_p(pv_dict, n)
            feature_list.append(dp.reshape(-1, 1))

        if "reflux_ratio" in self._cfg.enabled_groups:
            rr = self._compute_reflux_ratio(pv_dict, n)
            feature_list.append(rr.reshape(-1, 1))

        if "temperature_gradient" in self._cfg.enabled_groups:
            tg = self._compute_temperature_gradient(pv_dict, n)
            feature_list.append(tg.reshape(-1, 1))

        if "rolling_statistics" in self._cfg.enabled_groups:
            rs_feats = self._compute_rolling_stats(pv_dict, n)
            feature_list.append(rs_feats)

        if "interaction_terms" in self._cfg.enabled_groups:
            it_feats = self._compute_interaction_terms(pv_dict, n, dp=dp, rr=rr)
            feature_list.append(it_feats)

        if not feature_list:
            raise ValueError(
                "No feature groups enabled. Set FeatureConfig.enabled_groups "
                "to a non-empty sequence."
            )
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
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_length(pv_dict: Dict[str, np.ndarray]) -> int:
        """Return the consensus signal length from all present arrays."""
        lengths = {int(len(v)) for v in pv_dict.values() if hasattr(v, "__len__")}
        if not lengths:
            raise ValueError("pv_dict contains no finite-length arrays")
        if len(lengths) > 1:
            raise ValueError(
                f"All arrays in pv_dict must have the same length, got {sorted(lengths)}"
            )
        return lengths.pop()

    def _compute_delta_p(
        self, pv: Dict[str, np.ndarray], n: int
    ) -> np.ndarray:
        top = pv.get(self._cfg.top_pressure_tag)
        bot = pv.get(self._cfg.bottom_pressure_tag)
        if top is None or bot is None:
            return np.zeros(n)
        return bot - top

    def _compute_reflux_ratio(
        self, pv: Dict[str, np.ndarray], n: int
    ) -> np.ndarray:
        reflux = pv.get("FT-302", np.zeros(n))
        distillate = pv.get("FT-303", np.ones(n))
        distillate = np.where(np.abs(distillate) < 1e-6, 1e-6, distillate)
        return reflux / distillate

    def _compute_temperature_gradient(
        self, pv: Dict[str, np.ndarray], n: int
    ) -> np.ndarray:
        top_temp = pv.get("TE-301", np.zeros(n))
        bot_temp = pv.get("TE-310", np.zeros(n))
        return bot_temp - top_temp

    def _compute_rolling_stats(
        self, pv: Dict[str, np.ndarray], n: int
    ) -> np.ndarray:
        w = self._cfg.rolling_mean_window
        feats: List[np.ndarray] = []
        for tag in ["PT-301", "FT-301", "TE-301"]:
            arr = pv.get(tag, np.zeros(n))
            if len(arr) < w:
                mean_arr = np.full(n, float(np.mean(arr)))
                std_arr = np.full(n, float(np.std(arr)))
            else:
                mean_arr, std_arr = self._rolling_mean_std(arr, w)
            feats.extend([mean_arr.reshape(-1, 1), std_arr.reshape(-1, 1)])
        return np.hstack(feats)

    def _compute_interaction_terms(
        self,
        pv: Dict[str, np.ndarray],
        n: int,
        dp: np.ndarray | None = None,
        rr: np.ndarray | None = None,
    ) -> np.ndarray:
        # Reuse pre-computed dp / rr when available (avoids recomputation).
        if dp is None:
            dp = self._compute_delta_p(pv, n)
        if rr is None:
            rr = self._compute_reflux_ratio(pv, n)
        flow = pv.get("FT-301", np.ones(n))
        valve = pv.get("FV-301", np.ones(n))
        temp = pv.get("TE-301", np.ones(n))
        return np.column_stack([dp * rr, dp * flow, temp * valve])

    # ------------------------------------------------------------------
    # Efficient O(n) rolling statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_mean_std(
        arr: np.ndarray, window: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute rolling mean and std in O(n) using cumulative sums.

        Args:
            arr: 1-D input array, length >= window.
            window: rolling window size.

        Returns:
            (mean, std) — both 1-D arrays of same length as *arr*.
            The first ``window - 1`` positions replicate the first
            full-window value (forward-fill).
        """
        n = len(arr)
        out_mean = np.empty(n, dtype=np.float64)
        out_std = np.empty(n, dtype=np.float64)

        # Rolling mean via cumsum — O(n)
        cs = np.empty(n + 1, dtype=np.float64)
        cs[0] = 0.0
        np.cumsum(arr, out=cs[1:])
        windowed_mean = (cs[window:] - cs[:-window]) / window

        # Rolling std via cumsum of squares — O(n)
        cs2 = np.empty(n + 1, dtype=np.float64)
        cs2[0] = 0.0
        np.cumsum(arr.astype(np.float64) ** 2, out=cs2[1:])
        windowed_var = (cs2[window:] - cs2[:-window]) / window - windowed_mean**2
        windowed_var = np.maximum(windowed_var, 0.0)
        windowed_std = np.sqrt(windowed_var)

        # Forward-fill the leading edge
        out_mean[: window - 1] = windowed_mean[0]
        out_mean[window - 1 :] = windowed_mean
        out_std[: window - 1] = windowed_std[0]
        out_std[window - 1 :] = windowed_std

        return out_mean, out_std


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
