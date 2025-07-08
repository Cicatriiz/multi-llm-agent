"""Main AB-MCTS agent implementation that integrates with the multi-LLM system."""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.base import LLMAgent
from .ab_mcts_a import ABMCTSA, ABMCTSAAlgoState
from .ab_mcts_m import ABMCTSM, ABMCTSMState
from .ranker import top_k
from .types import GenerateFnType, StateScoreType

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM with content and score."""
    content: str
    model_name: str
    score: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ABMCTSAgent:
    """
    AB-MCTS Agent that uses adaptive branching Monte Carlo Tree Search
    to intelligently select and coordinate multiple LLM agents.
    """

    def __init__(
        self,
        agents: Dict[str, LLMAgent],
        algorithm: str = "ab_mcts_a",
        search_budget: int = 20,
        top_k_results: int = 1,
    ):
        """
        Initialize the AB-MCTS Agent.

        Args:
            agents: Dictionary mapping agent names to LLMAgent instances
            algorithm: Algorithm to use ("ab_mcts_a" or "ab_mcts_m")
            search_budget: Number of search iterations to perform
            top_k_results: Number of top results to return
        """
        self.agents = agents
        self.search_budget = search_budget
        self.top_k_results = top_k_results

        # Initialize the appropriate algorithm
        if algorithm == "ab_mcts_a":
            self.algo = ABMCTSA()
        elif algorithm == "ab_mcts_m":
            self.algo = ABMCTSM()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        logger.info(f"Initialized ABMCTSAgent with {algorithm} and {len(agents)} agents")

    def _create_generate_functions(
        self, prompt: str
    ) -> Dict[str, GenerateFnType[LLMResponse]]:
        """
        Create generation functions for each LLM agent.

        Args:
            prompt: The input prompt for generation

        Returns:
            Dictionary mapping agent names to generation functions
        """
        generate_fns = {}

        for agent_name, agent in self.agents.items():
            def make_generate_fn(agent_instance: LLMAgent, name: str):
                def generate_fn(parent_state: Optional[LLMResponse] = None) -> Tuple[LLMResponse, float]:
                    """Generate a response from this agent."""
                    try:
                        if parent_state is None:
                            # Initial generation
                            result = agent_instance.generate(prompt)
                        else:
                            # Refinement based on parent state
                            refinement_prompt = self._create_refinement_prompt(prompt, parent_state)
                            result = agent_instance.generate(refinement_prompt)

                        # Create response object
                        response = LLMResponse(
                            content=result.get("content", ""),
                            model_name=name,
                            score=self._score_response(result.get("content", "")),
                            metadata=result
                        )

                        return response, response.score

                    except Exception as e:
                        logger.error(f"Error generating from {name}: {e}")
                        # Return a low-score fallback response
                        return LLMResponse(
                            content=f"Error: {str(e)}",
                            model_name=name,
                            score=0.0
                        ), 0.0

                return generate_fn

            generate_fns[agent_name] = make_generate_fn(agent, agent_name)

        return generate_fns

    def _create_refinement_prompt(self, original_prompt: str, parent_state: LLMResponse) -> str:
        """
        Create a refinement prompt based on the parent state.

        Args:
            original_prompt: The original prompt
            parent_state: The parent response to refine

        Returns:
            Refined prompt string
        """
        return f"""Original prompt: {original_prompt}

Previous response (from {parent_state.model_name}, score: {parent_state.score:.2f}):
{parent_state.content}

Please provide an improved version of this response that addresses any limitations or enhances the quality."""

    def _score_response(self, content: str) -> float:
        """
        Score a response (simplified scoring function).

        Args:
            content: The response content to score

        Returns:
            Score between 0 and 1
        """
        if not content or content.startswith("Error:"):
            return 0.0

        # Simple heuristic scoring based on content characteristics
        score = 0.5  # base score

        # Reward longer, more detailed responses
        if len(content) > 100:
            score += 0.2
        if len(content) > 500:
            score += 0.1

        # Penalize very short responses
        if len(content) < 50:
            score -= 0.2

        # Simple quality indicators
        if any(word in content.lower() for word in ["therefore", "because", "however", "furthermore"]):
            score += 0.1

        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))

    def generate(self, prompt: str) -> Dict[str, Any]:
        """
        Generate responses using AB-MCTS to coordinate multiple LLM agents.

        Args:
            prompt: The input prompt

        Returns:
            Dictionary containing the best responses and search statistics
        """
        logger.info(f"Starting AB-MCTS generation with budget {self.search_budget}")

        # Create generation functions for each agent
        generate_fns = self._create_generate_functions(prompt)

        # Initialize the search tree
        search_state = self.algo.init_tree()

        # Run the search
        for i in range(self.search_budget):
            search_state = self.algo.step(search_state, generate_fns)
            
            if (i + 1) % 5 == 0:
                # Log intermediate progress
                try:
                    interim_results = top_k(search_state, self.algo, min(self.top_k_results, len(search_state.tree.get_state_score_pairs())))
                    best_score = interim_results[0][1] if interim_results else 0.0
                    logger.info(f"Iteration {i+1}: Best score so far = {best_score:.3f}")
                except Exception as e:
                    logger.debug(f"Could not get interim results: {e}")

        # Extract top-k results
        try:
            results = top_k(search_state, self.algo, min(self.top_k_results, len(search_state.tree.get_state_score_pairs())))
        except Exception as e:
            logger.error(f"Error extracting results: {e}")
            results = []

        # Format response
        if results:
            best_response = results[0][0]  # Best LLMResponse
            
            return {
                "content": best_response.content,
                "model": best_response.model_name,
                "score": best_response.score,
                "metadata": {
                    "algorithm": "ab_mcts",
                    "search_budget": self.search_budget,
                    "total_nodes": len(search_state.tree),
                    "all_results": [
                        {
                            "content": result[0].content,
                            "model": result[0].model_name,
                            "score": result[1]
                        }
                        for result in results
                    ]
                }
            }
        else:
            # Fallback if no results
            logger.warning("No results from AB-MCTS search, using fallback")
            fallback_agent = next(iter(self.agents.values()))
            fallback_result = fallback_agent.generate(prompt)
            
            return {
                "content": fallback_result.get("content", "No response generated"),
                "model": "fallback",
                "score": 0.0,
                "metadata": {
                    "algorithm": "ab_mcts",
                    "search_budget": self.search_budget,
                    "total_nodes": 0,
                    "fallback": True
                }
            }

    def get_search_stats(self, search_state: Union[ABMCTSAAlgoState, ABMCTSMState]) -> Dict[str, Any]:
        """
        Get statistics about the search process.

        Args:
            search_state: The final search state

        Returns:
            Dictionary with search statistics
        """
        stats = {
            "total_nodes": len(search_state.tree),
            "tree_depth": max((node.depth for node in search_state.tree.get_nodes()), default=0),
            "agents_used": list(self.agents.keys()),
        }

        if hasattr(search_state, "all_rewards_store"):
            # AB-MCTS-A specific stats
            stats["rewards_per_agent"] = {
                agent: len(rewards) for agent, rewards in search_state.all_rewards_store.items()
            }

        return stats
