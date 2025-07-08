"""Implementation of AB-MCTS-M algorithm for multi-LLM agents."""

import copy
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Dict, Generic, List, Optional, Tuple, TypeVar, Union

from .base import Algorithm
from .tree import Node, Tree
from .types import GenerateFnType, StateScoreType
from .imports import try_import

with try_import() as _import:
    import pymc as pm  # type: ignore
    from pymc.sampling.jax import sample_numpyro_nuts  # type: ignore

# Type variable for state
StateT = TypeVar("StateT")

@dataclass
class Observation:
    """Simple observation class for AB-MCTS-M."""
    reward: float
    action: str
    node_expand_idx: int
    
    @classmethod
    def collect_all_observations_of_descendant(
        cls,
        parent: Node,
        all_observations: Dict[int, "Observation"],
    ) -> List["Observation"]:
        """
        A helper method to collect all the descendant observations.
        """
        observations: List["Observation"] = []
        for child_idx, child in enumerate(parent.children):
            if child.expand_idx in all_observations:
                observations.append(all_observations[child.expand_idx])
                # Recursively collect from children
                observations.extend(cls.collect_all_observations_of_descendant(child, all_observations))
        return observations

@dataclass
class ABMCTSMState(Generic[StateT]):
    """State for ABMCTSM Algorithm."""

    tree: Tree[StateT]
    all_observations: Dict[int, Observation] = field(default_factory=dict)


class PyMCInterface:
    """
    Handles PyMC for Bayesian mixed models.
    """

    def __init__(self, enable_pruning=True, pruning_config=None, strategy="multiarm_bandit_thompson"):
        self.enable_pruning = enable_pruning
        self.pruning_config = pruning_config
        self.model_selection_strategy = strategy

    def run(
        self,
        observations: List[Observation],
        actions: List[str],
        node: Node,
        all_observations: List[Observation],
    ) -> Union[str, int]:
        """
        Main entry point of PyMC Bayesian mixed model fitting.
        """
        return "mock_action" if random.random() > 0.5 else 0  # Placeholder logic

class ABMCTSM(Algorithm[StateT, ABMCTSMState[StateT]]):
    """
    Monte Carlo Tree Search algorithm using AB-MCTS-M algorithm.
    """

    def __init__(
        self,
        enable_pruning: bool = True,
        reward_average_priors: Optional[Union[float, Dict[str, float]]] = None,
        model_selection_strategy: str = "multiarm_bandit_thompson",
        min_subtree_size_for_pruning: int = 4,
        same_score_proportion_threshold: float = 0.75,
    ):
        self.enable_pruning = enable_pruning
        self.reward_average_priors = reward_average_priors
        self.model_selection_strategy = model_selection_strategy
        
        # Create PyMC interface
        self.pymc_interface = PyMCInterface(
            enable_pruning=enable_pruning,
            strategy=model_selection_strategy
        )

    def init_tree(self) -> ABMCTSMState:
        """
        Initialize the algorithm state with an empty tree.
        """
        tree: Tree = Tree.with_root_node()
        return ABMCTSMState(tree=tree)

    def step(
        self,
        state: ABMCTSMState,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
        inplace: bool = False,
    ) -> ABMCTSMState:
        """
        Perform one step of AB-MCTS-M algorithm and generate one node.
        """
        if not inplace:
            state = copy.deepcopy(state)

        if not state.tree.root.children:
            self._expand_node(state, state.tree.root, generate_fn)
            return state

        node = state.tree.root

        while node.children:
            node, action = self._select_child(state, node, generate_fn)

            if action is not None:
                return state

        self._expand_node(state, node, generate_fn)

        return state

    def _select_child(
        self,
        state: ABMCTSMState,
        node: Node,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
    ) -> Tuple[Node, Optional[str]]:
        """
        Select a child node for expansion.
        """
        observations = Observation.collect_all_observations_of_descendant(
            node, state.all_observations
        )

        actions = list(generate_fn.keys())

        child_identifier = self.pymc_interface.run(
            observations,
            actions=actions,
            node=node,
            all_observations=list(state.all_observations.values()),
        )

        if isinstance(child_identifier, str):
            new_node = self._generate_new_child(state, node, generate_fn, child_identifier)
            return new_node, child_identifier
        else:
            return node.children[child_identifier], None

    def _expand_node(
        self,
        state: ABMCTSMState,
        node: Node,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
    ) -> Tuple[Node, str]:
        """
        Expand a leaf node by generating a new child.
        """
        observations = Observation.collect_all_observations_of_descendant(
            node, state.all_observations
        )

        actions = list(generate_fn.keys())

        node_identifier = self.pymc_interface.run(
            observations,
            actions=actions,
            node=node,
            all_observations=list(state.all_observations.values()),
        )

        # Ensure we get a string model name, not an index
        if not isinstance(node_identifier, str):
            raise ValueError(
                f"Internal Error: Expected model name string but got index {node_identifier}"
            )

        new_node = self._generate_new_child(state, node, generate_fn, node_identifier)
        return new_node, node_identifier

    def _generate_new_child(
        self,
        state: ABMCTSMState,
        node: Node,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
        action: str,
    ) -> Node:
        """
        Generate a new child node using the specified model.
        """
        # Generate new state and score using the selected model
        node_state = None if node.is_root() else node.state
        new_state, new_score = generate_fn[action](node_state)

        # Add new node to the tree
        new_node = state.tree.add_node((new_state, new_score), node)

        # Record observation
        state.all_observations[new_node.expand_idx] = Observation(
            reward=new_score, action=action, node_expand_idx=new_node.expand_idx
        )

        return new_node

    def get_state_score_pairs(
        self, state: ABMCTSMState
    ) -> List[StateScoreType[StateT]]:
        """
        Get all the state-score pairs from the tree.
        """
        return state.tree.get_state_score_pairs()
