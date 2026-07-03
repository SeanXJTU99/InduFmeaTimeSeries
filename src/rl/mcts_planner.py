"""MCTS (Monte Carlo Tree Search) planner for counterfactual fault simulation.

When the PLC detects a micro-perturbation (e.g. 0.5 °C temperature
swing), the MCTS planner explores possible future trajectories in the
digital twin sandbox.  Each node represents a system state; each edge
is a control action.  The search asks:

    "If valve FV-301 is sticky, what trajectory does the column follow?"
    vs.
    "If the cold-leak is at heat exchanger E-301, what trajectory?"

By running thousands of simulations, MCTS identifies the most likely
root cause *before* the async Excel abundance report arrives.

All parameters are fictitious.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MCTSConfig:
    """MCTS planner parameters."""

    n_simulations: int = 1000
    exploration_constant: float = 1.414  # UCB1 c parameter
    max_depth: int = 50
    gamma: float = 0.99  # discount factor


class MCTSNode:
    """A node in the MCTS search tree."""

    def __init__(
        self,
        state: Dict[str, float] | None = None,
        parent: "MCTSNode | None" = None,
        action: int | None = None,
    ) -> None:
        self.state = state or {}
        self.parent = parent
        self.action = action
        self.children: List[MCTSNode] = []
        self.visits: int = 0
        self.value: float = 0.0
        self.untried_actions: List[int] = list(range(5))  # 5 discrete actions

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def best_child(self, c: float = 1.414) -> "MCTSNode":
        """Select child with highest UCB1 score."""
        best_score = -float("inf")
        best: MCTSNode = self
        for child in self.children:
            if child.visits == 0:
                return child
            exploit = child.value / child.visits
            explore = c * math.sqrt(math.log(self.visits) / child.visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best = child
        return best

    def expand(self) -> "MCTSNode":
        """Add one child node for an untried action."""
        action = self.untried_actions.pop()
        child = MCTSNode(state=self.state.copy(), parent=self, action=action)
        self.children.append(child)
        return child

    def update(self, reward: float) -> None:
        """Backpropagate reward up the tree."""
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.update(reward)


class MCTSPlanner:
    """Monte Carlo Tree Search for fault-hypothesis evaluation.

    Usage::

        planner = MCTSPlanner(MCTSConfig(n_simulations=1000))
        root = MCTSNode(state={"dp": 1.8, "temp": -183.0, "flow": 48.0})
        best_action, stats = planner.search(root)
    """

    def __init__(self, config: MCTSConfig | None = None) -> None:
        self._cfg = config or MCTSConfig()

    def search(self, root: MCTSNode) -> Tuple[int, Dict[str, Any]]:
        """Run MCTS from *root* and return the best action.

        Args:
            root: initial state node.

        Returns:
            (best_action_index, statistics_dict).
        """
        for _ in range(self._cfg.n_simulations):
            node = self._select(root)
            if node.visits > 0 and not node.is_fully_expanded():
                node = node.expand()
            reward = self._simulate(node)
            node.update(reward)

        best = root.best_child(c=0.0)  # exploit only
        return (
            best.action if best.action is not None else 0,
            {
                "simulations": self._cfg.n_simulations,
                "root_visits": root.visits,
                "best_value": best.value / max(best.visits, 1),
                "best_visits": best.visits,
            },
        )

    def _select(self, node: MCTSNode) -> MCTSNode:
        depth = 0
        while node.is_fully_expanded() and node.children and depth < self._cfg.max_depth:
            node = node.best_child(self._cfg.exploration_constant)
            depth += 1
        return node

    def _simulate(self, node: MCTSNode) -> float:
        """Rollout: randomly walk until terminal or max depth, return cumulative reward."""
        total = 0.0
        state = node.state.copy()
        for _ in range(self._cfg.max_depth):
            action = random.randint(0, 4)
            r = self._step_reward(state, action)
            total += r * (self._cfg.gamma ** _)
            if r < -0.5:  # terminal (flooding / dry-bed)
                break
        return total

    @staticmethod
    def _step_reward(state: Dict[str, float], action: int) -> float:
        """Simplified reward for a single step (fictitious)."""
        dp = state.get("dp", 1.2)
        abundance = state.get("abundance", 97.0)
        # Penalise high ΔP (flooding risk), reward stable abundance
        if dp > 2.5:
            return -1.0
        if abundance < 95.0:
            return -0.5
        return 0.1
