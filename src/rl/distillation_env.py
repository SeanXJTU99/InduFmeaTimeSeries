"""Distillation column digital twin — OpenAI Gym environment.

A simplified mass-and-energy-balance model of a multi-stage cryogenic
distillation column.  The agent controls:
- Feed flow rate (FV position)
- Reflux ratio
- Reboiler duty

And observes:
- Top / bottom temperatures
- Pressure drop (ΔP)
- Isotope abundance (simulated with transport lag)

This environment is used by the PPO controller and MCTS planner to
learn optimal control policies that prevent flooding, dry-bed, and
cold-leak conditions BEFORE they show up in the async Excel abundance
report.

All parameters are fictitious — no real column is modelled.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces


@dataclass
class DistillationEnvConfig:
    """Digital twin parameters (fictitious column)."""

    # Column geometry
    n_stages: int = 30
    feed_stage: int = 15

    # Nominal operating point
    feed_flow_nominal: float = 50.0  # L/min
    reflux_ratio_nominal: float = 4.0
    reboiler_duty_nominal: float = 120.0  # kW
    top_temp_nominal: float = -185.0  # °C
    bottom_temp_nominal: float = -175.0  # °C
    dp_nominal: float = 1.2  # bar

    # Dynamics
    dt: float = 1.0  # simulation timestep (minutes)
    thermal_time_constant: float = 45.0  # minutes — lag to abundance response
    noise_std: float = 0.02  # process noise

    # Action space bounds (fraction of nominal)
    action_low: Tuple[float, ...] = (0.5, 0.5, 0.5)
    action_high: Tuple[float, ...] = (1.5, 1.5, 1.5)

    # Episode
    max_steps: int = 500


class DistillationEnv(gym.Env):
    """Cryogenic distillation column digital twin for RL training.

    Observation (8-D):
        [top_temp, bottom_temp, dp, feed_flow, reflux, reboiler_duty,
         abundance, abundance_rate]

    Action (3-D):
        [Δfeed_flow, Δreflux, Δreboiler] — normalised adjustments.

    Reward:
        +1.0  for keeping abundance in [95, 100]%
        -1.0  per flooding / dry-bed event
        -0.1  per step (efficiency incentive)
    """

    def __init__(self, config: DistillationEnvConfig | None = None) -> None:
        super().__init__()
        self._cfg = config or DistillationEnvConfig()
        self.action_space = spaces.Box(
            low=np.array(self._cfg.action_low, dtype=np.float32),
            high=np.array(self._cfg.action_high, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32,
        )
        self._state: Optional[np.ndarray] = None
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        # Start near nominal with small random perturbation
        self._state = np.array([
            self._cfg.top_temp_nominal + rng.normal(0, 0.5),
            self._cfg.bottom_temp_nominal + rng.normal(0, 0.5),
            self._cfg.dp_nominal + rng.normal(0, 0.05),
            self._cfg.feed_flow_nominal,
            self._cfg.reflux_ratio_nominal,
            self._cfg.reboiler_duty_nominal,
            97.0 + rng.normal(0, 0.3),  # abundance %
            0.0,  # abundance rate of change
        ], dtype=np.float32)
        self._step_count = 0
        return self._state.copy(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one control step.

        Args:
            action: [feed_mult, reflux_mult, reboiler_mult] multipliers.

        Returns:
            (observation, reward, terminated, truncated, info).
        """
        self._step_count += 1
        s = self._state.copy()
        cfg = self._cfg
        rng = np.random.default_rng()

        # Apply actions (first-order dynamics)
        feed = s[3] + cfg.dt * (action[0] * cfg.feed_flow_nominal - s[3]) / 10.0
        reflux = s[4] + cfg.dt * (action[1] * cfg.reflux_ratio_nominal - s[4]) / 10.0
        duty = s[5] + cfg.dt * (action[2] * cfg.reboiler_duty_nominal - s[5]) / 10.0

        # Simplified column model
        top_temp = (
            cfg.top_temp_nominal
            + 2.0 * (reflux / cfg.reflux_ratio_nominal - 1.0)
            - 1.5 * (duty / cfg.reboiler_duty_nominal - 1.0)
            + rng.normal(0, cfg.noise_std)
        )
        bot_temp = (
            cfg.bottom_temp_nominal
            + 3.0 * (duty / cfg.reboiler_duty_nominal - 1.0)
            + rng.normal(0, cfg.noise_std)
        )
        dp = (
            cfg.dp_nominal
            + 0.5 * (feed / cfg.feed_flow_nominal - 1.0)
            - 0.3 * (reflux / cfg.reflux_ratio_nominal - 1.0)
            + rng.normal(0, cfg.noise_std)
        )

        # Abundance — slow response (thermal time constant)
        target_abundance = 97.0 - 0.5 * (top_temp - cfg.top_temp_nominal)
        tau = cfg.thermal_time_constant
        abundance = s[6] + cfg.dt * (target_abundance - s[6]) / tau
        abundance += rng.normal(0, 0.1)
        abundance = float(np.clip(abundance, 0.0, 100.0))
        abundance_rate = (abundance - s[6]) / cfg.dt

        # Detect flooding / dry-bed
        flooding = dp > 3.0 * cfg.dp_nominal
        dry_bed = reflux < 0.3 * cfg.reflux_ratio_nominal

        # Reward
        reward = 0.0
        if 95.0 <= abundance <= 100.0:
            reward += 1.0
        if flooding or dry_bed:
            reward -= 1.0
        reward -= 0.1  # per-step penalty

        terminated = flooding or dry_bed
        truncated = self._step_count >= cfg.max_steps

        self._state = np.array([
            top_temp, bot_temp, dp, feed, reflux, duty,
            abundance, abundance_rate,
        ], dtype=np.float32)

        return self._state.copy(), reward, terminated, truncated, {
            "flooding": flooding,
            "dry_bed": dry_bed,
            "abundance": abundance,
        }
