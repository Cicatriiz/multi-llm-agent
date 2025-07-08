"""Implementation of AB-MCTS-A algorithm for multi-LLM agents."""

import copy
import random
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from logging import getLogger
from typing import Dict, Generic, List, Literal, Optional, Tuple, TypeVar, Union

from .base import Algorithm
from .tree import Node, Tree
from .types import GenerateFnType, StateScoreType

# Type variable for state
StateT = TypeVar("StateT")

logger = getLogger(__name__)

@dataclass
class ABMCTSAStateManager(Generic[StateT]):
    """Manager for ABMCTSA State instances associated with expand_idx values."""

    states: Dict[int, "NodeProbState"] = field(default_factory=dict)

    def __contains__(self, expand_idx: int) -> bool:
        """Check if a thompson state exists for the given expand_idx."""
        return expand_idx in self.states

    def __len__(self) -> int:
        return len(self.states)

    def get(self, node: Node) -> Optional["NodeProbState"]:
        """Get thompson state for the given expand_idx if it exists."""
        return self.states.get(node.expand_idx)

    def create(
        self,
        node: Node,
        actions: List[str],
    ) -> "NodeProbState":
        """Create a new thompson state for the given expand_idx with optional prior configuration."""
        state = NodeProbState(
            actions=actions,
        )
        self.states[node.expand_idx] = state
        return state

    def get_or_create(
        self,
        node: Node,
        actions: List[str],
    ) -> "NodeProbState":
        """Get existing thompson state or create a new one if it doesn't exist."""
        if node.expand_idx in self.states:
            return self.states[node.expand_idx]
        return self.create(
            node,
            actions,
        )

@dataclass
class ABMCTSAAlgoState(Generic[StateT]):
    """State for ABMCTSA algorithm."""

    tree: Tree[StateT]
    thompson_states: ABMCTSAStateManager = field(default_factory=ABMCTSAStateManager)
    all_rewards_store: Dict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )


class ABMCTSA(Algorithm[StateT, ABMCTSAAlgoState[StateT]]):
    """
    Adaptive Monte Carlo Tree Search algorithm with Node Aggregation.
    """

    def init_tree(self) -> ABMCTSAAlgoState:
        """
        Initialize the algorithm state with an empty tree.
        """
        tree: Tree = Tree.with_root_node()
        state_manager = ABMCTSAStateManager[StateT]()
        return ABMCTSAAlgoState(tree=tree, thompson_states=state_manager)

    def step(
        self,
        state: ABMCTSAAlgoState,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
        inplace: bool = False,
    ) -> ABMCTSAAlgoState:
        """
        Perform one step of the Thompson Sampling MCTS algorithm.
        """
        if not inplace:
            state = copy.deepcopy(state)

        if len(state.all_rewards_store) == 0:
            for action in generate_fn:
                state.all_rewards_store[action] = []

        if not state.tree.root.children:
            self._expand_node(state, state.tree.root, generate_fn)
            return state

        node = state.tree.root

        while node.children:
            node, action_used = self._select_child(state, node, generate_fn)

            if action_used is not None:
                return state

        self._expand_node(state, node, generate_fn)

        return state

    def _select_child(
        self,
        state: ABMCTSAAlgoState,
        node: Node,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
    ) -> Tuple[Node, Optional[str]]:
        """
        Select a child node using Thompson Sampling.
        """
        thompson_state = state.thompson_states.get_or_create(
            node,
            list(generate_fn.keys()),
        )

        selection = thompson_state.select_next(state.all_rewards_store)

        if isinstance(selection, str):
            new_node = self._generate_new_child(state, node, generate_fn, selection)
            return new_node, selection
        else:
            if selection >= len(node.children):
                raise RuntimeError(
                    f"Something went wrong in ABMCTSA algorithm: selected index {selection} is out of bounds."
                )
            return node.children[selection], None

    def _expand_node(
        self,
        state: ABMCTSAAlgoState,
        node: Node,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
    ) -> Tuple[Node, str]:
        """
        Expand a leaf node by generating a new child.
        """
        thompson_state = state.thompson_states.get_or_create(
            node,
            list(generate_fn.keys()),
        )

        selection = thompson_state.select_next(state.all_rewards_store)

        if not isinstance(selection, str):
            raise RuntimeError(
                f"Something went wrong in ABMCTSA algorithm: selection should always be str when the expansion is from the leaf node, while got {selection}"
            )

        new_node = self._generate_new_child(state, node, generate_fn, selection)
        return new_node, selection

    def _generate_new_child(
        self,
        state: ABMCTSAAlgoState,
        node: Node,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
        action: str,
    ) -> Node:
        """
        Generate a new child node using the specified action.
        """
        node_state = None if node.is_root() else node.state
        new_state, new_score = generate_fn[action](node_state)

        new_node = state.tree.add_node((new_state, new_score), node)

        thompson_state = state.thompson_states.get(node)
        if thompson_state:
            thompson_state.register_new_child_node(
                action, new_node
            )
        else:
            raise RuntimeError(
                f"Internal Error in ABMCTSA: thompson_state should not be None for node {node}"
            )

        self._backpropagate(state, new_node, new_score, action)

        return new_node

    def _backpropagate(
        self, state: ABMCTSAAlgoState, node: Node, score: float, action: str
    ) -> None:
        """
        Update Thompson Sampling statistics for all nodes in the path from node to root.
        """
        state.all_rewards_store[action].append(score)

        assert node.parent is not None
        thompson_state = state.thompson_states.get(node.parent)
        if thompson_state is None:
            raise RuntimeError(
                "Internal Error in ABMCTSA: ThompsonState should have been already initialized"
            )
        thompson_state.update_action_reward(action=action, reward=score)

        current: Optional[Node] = node.parent
        while current is not None and current.parent is not None:
            thompson_state = state.thompson_states.get(current.parent)
            if thompson_state is None:
                raise RuntimeError(
                    "Internal Error in ABMCTSA: ThompsonState should have been already initialized"
                )

            thompson_state.update_node_reward(current, score)

            current = current.parent

    def get_state_score_pairs(
        self, state: ABMCTSAAlgoState
    ) -> List[StateScoreType[StateT]]:
        """
        Get all the state-score pairs from the tree.
        """
        return state.tree.get_state_score_pairs()

class NodeProbState:
    """Probabilistic state for each node in ABMCTSA."""

    def __init__(self, actions: List[str]):
        self.actions = actions
        self.action_rewards: Dict[str, List[float]] = {action: [] for action in actions}
        self.child_nodes: List[Node] = []
        self.child_rewards: List[List[float]] = []
        self.action_to_nodes: Dict[str, List[int]] = {action: [] for action in actions}
    
    def select_next(self, rewards_store: Dict[str, List[float]]) -> Union[str, int]:
        """
        Use Thompson Sampling to select the next action or existing child.
        """
        # Simple logic: 50% chance to generate new vs continue with existing
        if not self.child_nodes or random.random() < 0.5:
            # Generate new node - select action based on rewards
            if all(len(rewards) == 0 for rewards in rewards_store.values()):
                return random.choice(self.actions)
            
            # Use simple Thompson sampling - select action with highest average + noise
            best_action = None
            best_score = -1
            for action in self.actions:
                if action in rewards_store and rewards_store[action]:
                    avg_reward = sum(rewards_store[action]) / len(rewards_store[action])
                    noise = random.gauss(0, 0.1)  # Add Gaussian noise
                    score = avg_reward + noise
                else:
                    score = random.random()  # Random score for unexplored actions
                
                if score > best_score:
                    best_score = score
                    best_action = action
            
            return best_action or random.choice(self.actions)
        else:
            # Continue with existing child
            return random.randint(0, len(self.child_nodes) - 1)

    def register_new_child_node(self, action: str, node: Node) -> None:
        """Register new child node under the given action."""
        self.child_nodes.append(node)
        self.child_rewards.append([node.score])
        if action in self.action_to_nodes:
            self.action_to_nodes[action].append(len(self.child_nodes) - 1)

    def update_action_reward(self, action: str, reward: float) -> None:
        """Update reward for given action."""
        if action in self.action_rewards:
            self.action_rewards[action].append(reward)

    def update_node_reward(self, node: Node, reward: float) -> None:
        """Update reward for given node."""
        # Find the node in our child list and update its rewards
        for i, child in enumerate(self.child_nodes):
            if child.expand_idx == node.expand_idx:
                self.child_rewards[i].append(reward)
                break

