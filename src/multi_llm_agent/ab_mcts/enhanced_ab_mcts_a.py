"""Enhanced AB-MCTS-A implementation with improved Thompson Sampling and exploration."""

import copy
import math
import random
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union

try:
    import numpy as np
except ImportError:
    # Fallback if numpy not available
    class FakeNumpy:
        @staticmethod
        def random_beta(alpha, beta_param):
            # Simple approximation using random.random()
            return random.random()
        
        @staticmethod
        def argmax(arr):
            return max(range(len(arr)), key=lambda i: arr[i])
    
    np = FakeNumpy()

try:
    from scipy.stats import beta, norm
except ImportError:
    # Scipy not available, we'll use simpler implementations
    pass

from .base import Algorithm
from .tree import Node, Tree
from .types import GenerateFnType, StateScoreType

# Type variable for state
StateT = TypeVar("StateT")

logger = getLogger(__name__)


@dataclass
class BayesianStats:
    """Bayesian statistics for Thompson Sampling."""
    alpha: float = 1.0  # Beta distribution parameter (successes + 1)
    beta_param: float = 1.0  # Beta distribution parameter (failures + 1)
    total_samples: int = 0
    sum_rewards: float = 0.0
    sum_squared_rewards: float = 0.0
    
    def update(self, reward: float):
        """Update statistics with a new reward."""
        self.total_samples += 1
        self.sum_rewards += reward
        self.sum_squared_rewards += reward * reward
        
        # Update Beta distribution parameters
        # Treat reward as success probability
        self.alpha += reward
        self.beta_param += (1.0 - reward)
    
    def sample(self) -> float:
        """Sample from the posterior distribution."""
        if self.total_samples == 0:
            return random.random()
        
        # Use Beta distribution for bounded rewards [0,1]
        try:
            # Try numpy version if available
            if hasattr(np, 'random') and hasattr(np.random, 'beta'):
                return np.random.beta(self.alpha, self.beta_param)
            else:
                # Fallback to simple approximation
                mean = self.alpha / (self.alpha + self.beta_param)
                variance = (self.alpha * self.beta_param) / ((self.alpha + self.beta_param) ** 2 * (self.alpha + self.beta_param + 1))
                # Add some noise around the mean
                return max(0.0, min(1.0, mean + random.gauss(0, math.sqrt(variance))))
        except:
            # Final fallback
            return self.mean + random.gauss(0, 0.1)
    
    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Get confidence interval for the mean."""
        if self.total_samples < 2:
            return (0.0, 1.0)
        
        mean = self.sum_rewards / self.total_samples
        variance = (self.sum_squared_rewards / self.total_samples) - (mean ** 2)
        std = math.sqrt(variance / self.total_samples)
        
        z_score = 1.96 if confidence == 0.95 else 2.576  # 99% confidence
        margin = z_score * std
        
        return (max(0.0, mean - margin), min(1.0, mean + margin))
    
    @property
    def mean(self) -> float:
        """Get the empirical mean."""
        return self.sum_rewards / self.total_samples if self.total_samples > 0 else 0.5
    
    @property
    def variance(self) -> float:
        """Get the empirical variance."""
        if self.total_samples < 2:
            return 0.25  # Default variance for [0,1] uniform
        mean = self.mean
        return (self.sum_squared_rewards / self.total_samples) - (mean ** 2)


class EnhancedNodeProbState:
    """Enhanced probabilistic state with better Thompson Sampling."""

    def __init__(self, actions: List[str], exploration_factor: float = 2.0):
        self.actions = actions
        self.exploration_factor = exploration_factor
        
        # Bayesian statistics for each action
        self.action_stats: Dict[str, BayesianStats] = {
            action: BayesianStats() for action in actions
        }
        
        # Statistics for GEN vs CONT decision
        self.gen_stats = BayesianStats()
        self.cont_stats = BayesianStats()
        
        # Child nodes and their statistics
        self.child_nodes: List[Node] = []
        self.child_stats: List[BayesianStats] = []
        self.action_to_nodes: Dict[str, List[int]] = {action: [] for action in actions}
        
        # UCB parameters
        self.total_visits = 0
    
    def select_next(self, rewards_store: Dict[str, List[float]]) -> Union[str, int]:
        """Enhanced selection using Thompson Sampling with UCB fallback."""
        self.total_visits += 1
        
        # Step 1: Decide between GEN (new node) vs CONT (existing child)
        if not self.child_nodes:
            # No children, must generate
            return self._select_best_action_thompson(rewards_store)
        
        # Use Thompson Sampling for GEN vs CONT decision
        gen_score = self.gen_stats.sample()
        cont_score = self.cont_stats.sample()
        
        # Add exploration bonus
        gen_exploration = self._ucb_exploration_bonus(self.gen_stats.total_samples)
        cont_exploration = self._ucb_exploration_bonus(self.cont_stats.total_samples)
        
        if gen_score + gen_exploration > cont_score + cont_exploration:
            # Generate new node
            return self._select_best_action_thompson(rewards_store)
        else:
            # Continue with existing child
            return self._select_best_child_thompson()
    
    def _select_best_action_thompson(self, rewards_store: Dict[str, List[float]]) -> str:
        """Select best action using Thompson Sampling."""
        action_scores = {}
        
        for action in self.actions:
            # Get Thompson sample
            thompson_score = self.action_stats[action].sample()
            
            # Add global information from rewards_store
            global_bonus = 0.0
            if action in rewards_store and rewards_store[action]:
                global_mean = sum(rewards_store[action]) / len(rewards_store[action])
                global_bonus = 0.1 * global_mean  # Weight global information
            
            # Add exploration bonus
            exploration_bonus = self._ucb_exploration_bonus(self.action_stats[action].total_samples)
            
            action_scores[action] = thompson_score + global_bonus + exploration_bonus
        
        return max(action_scores, key=action_scores.get)
    
    def _select_best_child_thompson(self) -> int:
        """Select best child using Thompson Sampling."""
        if not self.child_stats:
            return 0
        
        child_scores = []
        for i, stats in enumerate(self.child_stats):
            thompson_score = stats.sample()
            exploration_bonus = self._ucb_exploration_bonus(stats.total_samples)
            child_scores.append(thompson_score + exploration_bonus)
        
        try:
            if hasattr(np, 'argmax'):
                return int(np.argmax(child_scores))
            else:
                return max(range(len(child_scores)), key=lambda i: child_scores[i])
        except:
            return max(range(len(child_scores)), key=lambda i: child_scores[i])
    
    def _ucb_exploration_bonus(self, visits: int) -> float:
        """Calculate UCB exploration bonus."""
        if visits == 0:
            return float('inf')
        return self.exploration_factor * math.sqrt(math.log(self.total_visits) / visits)
    
    def register_new_child_node(self, action: str, node: Node) -> None:
        """Register new child node under the given action."""
        self.child_nodes.append(node)
        self.child_stats.append(BayesianStats())
        self.child_stats[-1].update(node.score)  # Initialize with node's score
        
        if action in self.action_to_nodes:
            self.action_to_nodes[action].append(len(self.child_nodes) - 1)
    
    def update_action_reward(self, action: str, reward: float) -> None:
        """Update reward for given action."""
        if action in self.action_stats:
            self.action_stats[action].update(reward)
        
        # Update GEN stats since we generated a new node
        self.gen_stats.update(reward)
    
    def update_node_reward(self, node: Node, reward: float) -> None:
        """Update reward for given node."""
        # Find the node in our child list and update its stats
        for i, child in enumerate(self.child_nodes):
            if child.expand_idx == node.expand_idx:
                self.child_stats[i].update(reward)
                break
        
        # Update CONT stats since we continued with an existing node
        self.cont_stats.update(reward)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about the current state."""
        return {
            "total_visits": self.total_visits,
            "num_children": len(self.child_nodes),
            "action_stats": {
                action: {
                    "mean": stats.mean,
                    "samples": stats.total_samples,
                    "confidence_interval": stats.confidence_interval()
                }
                for action, stats in self.action_stats.items()
            },
            "gen_cont_stats": {
                "gen_mean": self.gen_stats.mean,
                "gen_samples": self.gen_stats.total_samples,
                "cont_mean": self.cont_stats.mean,
                "cont_samples": self.cont_stats.total_samples,
            }
        }


@dataclass
class EnhancedABMCTSAStateManager(Generic[StateT]):
    """Enhanced manager for ABMCTSA State instances."""

    states: Dict[int, EnhancedNodeProbState] = field(default_factory=dict)
    exploration_factor: float = 2.0

    def get_or_create(self, node: Node, actions: List[str]) -> EnhancedNodeProbState:
        """Get existing state or create a new one if it doesn't exist."""
        if node.expand_idx not in self.states:
            self.states[node.expand_idx] = EnhancedNodeProbState(
                actions, exploration_factor=self.exploration_factor
            )
        return self.states[node.expand_idx]


@dataclass
class EnhancedABMCTSAAlgoState(Generic[StateT]):
    """Enhanced state for ABMCTSA algorithm."""

    tree: Tree[StateT]
    thompson_states: EnhancedABMCTSAStateManager = field(
        default_factory=EnhancedABMCTSAStateManager
    )
    all_rewards_store: Dict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # Performance tracking
    step_count: int = 0
    quality_progression: List[float] = field(default_factory=list)


class EnhancedABMCTSA(Algorithm[StateT, EnhancedABMCTSAAlgoState[StateT]]):
    """
    Enhanced Adaptive Monte Carlo Tree Search algorithm with improved Thompson Sampling.
    """

    def __init__(self, exploration_factor: float = 2.0):
        """
        Initialize the Enhanced AB-MCTS-A algorithm.
        
        Args:
            exploration_factor: UCB exploration parameter (higher = more exploration)
        """
        self.exploration_factor = exploration_factor

    def init_tree(self) -> EnhancedABMCTSAAlgoState:
        """Initialize the algorithm state with an empty tree."""
        tree: Tree = Tree.with_root_node()
        state_manager = EnhancedABMCTSAStateManager[StateT](
            exploration_factor=self.exploration_factor
        )
        return EnhancedABMCTSAAlgoState(tree=tree, thompson_states=state_manager)

    def step(
        self,
        state: EnhancedABMCTSAAlgoState,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
        inplace: bool = False,
    ) -> EnhancedABMCTSAAlgoState:
        """Perform one step of the Enhanced Thompson Sampling MCTS algorithm."""
        if not inplace:
            state = copy.deepcopy(state)

        state.step_count += 1

        # Initialize rewards store
        if len(state.all_rewards_store) == 0:
            for action in generate_fn:
                state.all_rewards_store[action] = []

        # If the tree is empty (only root), expand the root
        if not state.tree.root.children:
            self._expand_node(state, state.tree.root, generate_fn)
            return state

        # Run one simulation step
        node = state.tree.root

        # Selection phase: traverse tree until we reach a leaf node
        while node.children:
            node, action_used = self._select_child(state, node, generate_fn)

            # If action is not None, it means we've generated a new node
            if action_used is not None:
                # Track quality progression
                current_best = max(
                    state.tree.get_state_score_pairs(), key=lambda x: x[1], default=(None, 0.0)
                )[1]
                state.quality_progression.append(current_best)
                return state

        # Expansion phase: expand leaf node
        self._expand_node(state, node, generate_fn)
        
        # Track quality progression
        current_best = max(
            state.tree.get_state_score_pairs(), key=lambda x: x[1], default=(None, 0.0)
        )[1]
        state.quality_progression.append(current_best)

        return state

    def _select_child(
        self,
        state: EnhancedABMCTSAAlgoState,
        node: Node,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
    ) -> Tuple[Node, Optional[str]]:
        """Select a child node using Enhanced Thompson Sampling."""
        thompson_state = state.thompson_states.get_or_create(
            node, list(generate_fn.keys())
        )

        selection = thompson_state.select_next(state.all_rewards_store)

        if isinstance(selection, str):
            new_node = self._generate_new_child(state, node, generate_fn, selection)
            return new_node, selection
        else:
            if selection >= len(node.children):
                raise RuntimeError(
                    f"Selection index {selection} out of bounds for {len(node.children)} children"
                )
            return node.children[selection], None

    def _expand_node(
        self,
        state: EnhancedABMCTSAAlgoState,
        node: Node,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
    ) -> Tuple[Node, str]:
        """Expand a leaf node by generating a new child."""
        thompson_state = state.thompson_states.get_or_create(
            node, list(generate_fn.keys())
        )

        selection = thompson_state.select_next(state.all_rewards_store)

        if not isinstance(selection, str):
            raise RuntimeError(
                f"Selection should be string action for leaf expansion, got {selection}"
            )

        new_node = self._generate_new_child(state, node, generate_fn, selection)
        return new_node, selection

    def _generate_new_child(
        self,
        state: EnhancedABMCTSAAlgoState,
        node: Node,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
        action: str,
    ) -> Node:
        """Generate a new child node using the specified action."""
        node_state = None if node.is_root() else node.state
        new_state, new_score = generate_fn[action](node_state)

        new_node = state.tree.add_node((new_state, new_score), node)

        thompson_state = state.thompson_states.states.get(node.expand_idx)
        if thompson_state:
            thompson_state.register_new_child_node(action, new_node)
        else:
            raise RuntimeError(
                f"Thompson state should exist for node {node.expand_idx}"
            )

        self._backpropagate(state, new_node, new_score, action)

        return new_node

    def _backpropagate(
        self, state: EnhancedABMCTSAAlgoState, node: Node, score: float, action: str
    ) -> None:
        """Update Enhanced Thompson Sampling statistics for all nodes in the path."""
        state.all_rewards_store[action].append(score)

        assert node.parent is not None
        thompson_state = state.thompson_states.states.get(node.parent.expand_idx)
        if thompson_state is None:
            raise RuntimeError(
                "Thompson state should exist for parent node"
            )
        thompson_state.update_action_reward(action=action, reward=score)

        current: Optional[Node] = node.parent
        while current is not None and current.parent is not None:
            thompson_state = state.thompson_states.states.get(current.parent.expand_idx)
            if thompson_state is None:
                raise RuntimeError(
                    "Thompson state should exist for ancestor node"
                )

            thompson_state.update_node_reward(current, score)
            current = current.parent

    def get_state_score_pairs(
        self, state: EnhancedABMCTSAAlgoState
    ) -> List[StateScoreType[StateT]]:
        """Get all the state-score pairs from the tree."""
        return state.tree.get_state_score_pairs()
    
    def get_diagnostics(self, state: EnhancedABMCTSAAlgoState) -> Dict[str, Any]:
        """Get detailed diagnostics about the algorithm state."""
        diagnostics = {
            "step_count": state.step_count,
            "tree_size": len(state.tree),
            "quality_progression": state.quality_progression,
            "rewards_store": {k: len(v) for k, v in state.all_rewards_store.items()},
        }
        
        # Add node-level diagnostics for interesting nodes
        if state.thompson_states.states:
            sample_node_idx = list(state.thompson_states.states.keys())[0]
            sample_state = state.thompson_states.states[sample_node_idx]
            diagnostics["sample_node_diagnostics"] = sample_state.get_diagnostics()
        
        return diagnostics
