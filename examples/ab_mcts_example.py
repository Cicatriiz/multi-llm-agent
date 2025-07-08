#!/usr/bin/env python3
"""
Example demonstrating the use of AB-MCTS (Adaptive Branching Monte Carlo Tree Search)
with multiple LLM agents.
"""

import random
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock LLMAgent for demonstration
class MockLLMAgent:
    def __init__(self, name: str, quality_factor: float = 1.0):
        self.name = name
        self.quality_factor = quality_factor
    
    def generate(self, prompt: str) -> Dict[str, Any]:
        """Mock generation function."""
        # Simulate different response qualities based on the agent
        base_responses = [
            f"Based on the prompt '{prompt}', I would suggest...",
            f"After analyzing '{prompt}', my recommendation is...",
            f"Considering the context of '{prompt}', I propose...",
            f"To address '{prompt}', I believe the best approach is...",
        ]
        
        content = random.choice(base_responses) + f" [Response from {self.name}]"
        
        # Add some variation based on quality factor
        if random.random() < self.quality_factor:
            content += " This solution is well-researched and comprehensive."
        
        return {
            "content": content,
            "model": self.name,
            "usage": {"tokens": len(content.split())},
        }

def main():
    """Demonstrate AB-MCTS with mock agents."""
    
    # Import AB-MCTS components
    try:
        from multi_llm_agent.ab_mcts import ABMCTSAgent
        logger.info("Successfully imported AB-MCTS components")
    except ImportError as e:
        logger.error(f"Failed to import AB-MCTS: {e}")
        return
    
    # Create mock agents with different quality characteristics
    agents = {
        "gpt-4": MockLLMAgent("gpt-4", quality_factor=0.9),
        "claude-3": MockLLMAgent("claude-3", quality_factor=0.85),
        "gemini-pro": MockLLMAgent("gemini-pro", quality_factor=0.8),
        "local-llm": MockLLMAgent("local-llm", quality_factor=0.7),
    }
    
    # Test prompts
    test_prompts = [
        "Explain the concept of machine learning",
        "Write a Python function to sort a list",
        "Describe the impact of climate change",
        "How to design a scalable web API?",
    ]
    
    for algorithm in ["ab_mcts_a", "ab_mcts_m"]:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing {algorithm.upper()}")
        logger.info(f"{'='*50}")
        
        try:
            # Initialize AB-MCTS Agent
            ab_mcts_agent = ABMCTSAgent(
                agents=agents,
                algorithm=algorithm,
                search_budget=15,  # Number of search iterations
                top_k_results=3,   # Return top 3 results
            )
            
            for prompt in test_prompts:
                logger.info(f"\nPrompt: {prompt}")
                logger.info("-" * 40)
                
                # Generate response using AB-MCTS
                result = ab_mcts_agent.generate(prompt)
                
                # Display results
                logger.info(f"Best Response (Score: {result['score']:.3f}):")
                logger.info(f"  Model: {result['model']}")
                logger.info(f"  Content: {result['content'][:100]}...")
                
                # Show metadata
                metadata = result.get('metadata', {})
                logger.info(f"  Search Statistics:")
                logger.info(f"    - Total nodes explored: {metadata.get('total_nodes', 'N/A')}")
                logger.info(f"    - Search budget used: {metadata.get('search_budget', 'N/A')}")
                
                if 'all_results' in metadata:
                    logger.info(f"    - Top {len(metadata['all_results'])} results:")
                    for i, res in enumerate(metadata['all_results']):
                        logger.info(f"      {i+1}. {res['model']} (score: {res['score']:.3f})")
                
        except Exception as e:
            logger.error(f"Error testing {algorithm}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
