"""Tests for distillation RL environment and PPO controller."""

import numpy as np

from src.rl.distillation_env import DistillationEnv, DistillationEnvConfig
from src.rl.ppo_controller import PPOController, PPOPolicy


class TestDistillationEnv:
    def test_reset_returns_valid_obs(self) -> None:
        env = DistillationEnv()
        obs, info = env.reset(seed=42)
        assert obs.shape == (8,)
        assert isinstance(info, dict)

    def test_step_returns_expected_shapes(self) -> None:
        env = DistillationEnv()
        env.reset(seed=42)
        action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (8,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_episode_terminates(self) -> None:
        env = DistillationEnv(DistillationEnvConfig(max_steps=10))
        env.reset(seed=42)
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(np.array([1.0, 1.0, 1.0], dtype=np.float32))
            if terminated or truncated:
                break
        assert terminated or truncated


class TestPPOPolicy:
    def test_output_shapes(self) -> None:
        policy = PPOPolicy(obs_dim=8, act_dim=3, hidden_dim=128)
        import torch
        obs = torch.randn(4, 8)
        mean, value = policy(obs)  # type: ignore[arg-type]
        assert mean.shape == (4, 3)
        assert value.shape == (4, 1)

    def test_get_action(self) -> None:
        policy = PPOPolicy(obs_dim=8, act_dim=3)
        import torch
        obs = torch.randn(1, 8)
        action, log_prob, value = policy.get_action(obs)  # type: ignore[arg-type]
        assert action.shape == (1, 3)
        assert log_prob.shape == (1,)
        assert value.shape == (1,)
