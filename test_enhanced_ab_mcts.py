#!/usr/bin/env python3
"""
Test comparing original vs enhanced AB-MCTS implementations.
"""

import sys
import os
import random
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
import statistics

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class TestAgent:
    """Agent with configurable behavior for testing."""
    
    def __init__(self, name: str, quality: float = 0.8, response_time: float = 0.01):
        self.name = name
        self.quality = quality
        self.response_time = response_time
        self.call_count = 0
        
    def generate(self, prompt: str) -> Dict[str, Any]:
        """Generate a test response."""
        self.call_count += 1
        time.sleep(self.response_time)  # Simulate processing time
        
        base_score = self.quality + random.gauss(0, 0.1)  # Add noise
        score = max(0.0, min(1.0, base_score))  # Clamp to [0,1]
        
        content = f"Response from {self.name} for '{prompt[:30]}...'"
        if score > 0.8:
            content += " High quality response with detailed analysis."
        
        return {
            "content": content,
            "model": self.name,
            "quality": score,
        }

def create_test_agents() -> Dict[str, TestAgent]:
    """Create diverse test agents."""
    return {
        "high_quality": TestAgent("high_quality", quality=0.9, response_time=0.02),
        "medium_quality": TestAgent("medium_quality", quality=0.7, response_time=0.01),
        "fast_low_quality": TestAgent("fast_low_quality", quality=0.5, response_time=0.005),
        "slow_high_quality": TestAgent("slow_high_quality", quality=0.95, response_time=0.03),
    }

def run_algorithm_comparison(num_iterations: int = 30):
    """Compare original vs enhanced AB-MCTS-A implementations."""
    
    logger.info(f"🔬 Comparing Original vs Enhanced AB-MCTS-A ({num_iterations} iterations)")
    logger.info("=" * 70)
    
    try:
        # Import both implementations
        from multi_llm_agent.ab_mcts import ABMCTSA as OriginalABMCTSA
        from multi_llm_agent.ab_mcts.enhanced_ab_mcts_a import EnhancedABMCTSA
        from multi_llm_agent.ab_mcts import top_k
        from multi_llm_agent.ab_mcts.algorithm import LLMResponse
        
        agents = create_test_agents()
        
        # Create generation functions
        def create_generate_fn(agent, name):
            def generate_fn(parent_state=None):
                prompt = "Solve this complex optimization problem"
                if parent_state:
                    prompt = f"Improve on: {parent_state.content[:30]}..."
                
                result = agent.generate(prompt)
                response = LLMResponse(
                    content=result["content"],
                    model_name=name,
                    score=result["quality"],
                    metadata=result
                )
                return response, response.score
            return generate_fn
        
        generate_fns = {name: create_generate_fn(agent, name) for name, agent in agents.items()}
        
        results = {}
        
        # Test Original AB-MCTS-A
        logger.info("🧪 Testing Original AB-MCTS-A...")
        start_time = time.time()
        
        orig_algo = OriginalABMCTSA()
        orig_state = orig_algo.init_tree()
        
        orig_scores = []
        for i in range(num_iterations):
            orig_state = orig_algo.step(orig_state, generate_fns)
            if orig_state.tree.get_state_score_pairs():
                best_score = max(pair[1] for pair in orig_state.tree.get_state_score_pairs())
                orig_scores.append(best_score)
        
        orig_time = time.time() - start_time
        orig_results = top_k(orig_state, orig_algo, k=5)
        
        results["original"] = {
            "time": orig_time,
            "best_score": orig_results[0][1] if orig_results else 0.0,
            "avg_score": statistics.mean([r[1] for r in orig_results]) if orig_results else 0.0,
            "score_progression": orig_scores,
            "tree_size": len(orig_state.tree),
            "agent_calls": {name: agent.call_count for name, agent in agents.items()}
        }
        
        # Reset agent call counts
        for agent in agents.values():
            agent.call_count = 0
        
        # Test Enhanced AB-MCTS-A
        logger.info("🧪 Testing Enhanced AB-MCTS-A...")
        start_time = time.time()
        
        enhanced_algo = EnhancedABMCTSA(exploration_factor=1.5)
        enhanced_state = enhanced_algo.init_tree()
        
        enhanced_scores = []
        for i in range(num_iterations):
            enhanced_state = enhanced_algo.step(enhanced_state, generate_fns)
            if enhanced_state.tree.get_state_score_pairs():
                best_score = max(pair[1] for pair in enhanced_state.tree.get_state_score_pairs())
                enhanced_scores.append(best_score)
        
        enhanced_time = time.time() - start_time
        enhanced_results = top_k(enhanced_state, enhanced_algo, k=5)
        enhanced_diagnostics = enhanced_algo.get_diagnostics(enhanced_state)
        
        results["enhanced"] = {
            "time": enhanced_time,
            "best_score": enhanced_results[0][1] if enhanced_results else 0.0,
            "avg_score": statistics.mean([r[1] for r in enhanced_results]) if enhanced_results else 0.0,
            "score_progression": enhanced_scores,
            "tree_size": len(enhanced_state.tree),
            "agent_calls": {name: agent.call_count for name, agent in agents.items()},
            "diagnostics": enhanced_diagnostics
        }
        
        # Analysis
        logger.info("\n📊 COMPARISON RESULTS")
        logger.info("=" * 50)
        
        for name, result in results.items():
            logger.info(f"\n{name.upper()} AB-MCTS-A:")
            logger.info(f"  • Best Score: {result['best_score']:.4f}")
            logger.info(f"  • Average Score: {result['avg_score']:.4f}")
            logger.info(f"  • Runtime: {result['time']:.3f}s")
            logger.info(f"  • Tree Size: {result['tree_size']} nodes")
            logger.info(f"  • Total Agent Calls: {sum(result['agent_calls'].values())}")
            
            # Most used agent
            most_used = max(result['agent_calls'].items(), key=lambda x: x[1])
            logger.info(f"  • Most Used Agent: {most_used[0]} ({most_used[1]} calls)")
            
            # Score progression analysis
            if result['score_progression']:
                final_score = result['score_progression'][-1]
                initial_score = result['score_progression'][0]
                improvement = final_score - initial_score
                logger.info(f"  • Score Improvement: +{improvement:.4f}")
        
        # Winner analysis
        if len(results) == 2:
            orig = results["original"]
            enh = results["enhanced"]
            
            logger.info(f"\n🏆 WINNER ANALYSIS:")
            logger.info(f"  • Better Score: {'Enhanced' if enh['best_score'] > orig['best_score'] else 'Original'}")
            logger.info(f"  • Faster: {'Enhanced' if enh['time'] < orig['time'] else 'Original'}")
            logger.info(f"  • More Efficient: {'Enhanced' if sum(enh['agent_calls'].values()) < sum(orig['agent_calls'].values()) else 'Original'}")
            
            # Score progression comparison
            if orig['score_progression'] and enh['score_progression']:
                orig_final_improvement = orig['score_progression'][-1] - orig['score_progression'][0]
                enh_final_improvement = enh['score_progression'][-1] - enh['score_progression'][0]
                logger.info(f"  • Better Learning: {'Enhanced' if enh_final_improvement > orig_final_improvement else 'Original'}")
        
        # Enhanced-specific diagnostics
        if "diagnostics" in results["enhanced"]:
            diag = results["enhanced"]["diagnostics"]
            logger.info(f"\n🔍 ENHANCED ALGORITHM DIAGNOSTICS:")
            logger.info(f"  • Quality Progression Length: {len(diag.get('quality_progression', []))}")
            if "sample_node_diagnostics" in diag:
                sample_diag = diag["sample_node_diagnostics"]
                logger.info(f"  • Sample Node Visits: {sample_diag.get('total_visits', 'N/A')}")
                logger.info(f"  • Sample Node Children: {sample_diag.get('num_children', 'N/A')}")
        
        return results
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

def test_exploration_factors():
    """Test different exploration factors for Enhanced AB-MCTS-A."""
    
    logger.info("\n🔬 Testing Exploration Factors")
    logger.info("=" * 50)
    
    try:
        from multi_llm_agent.ab_mcts.enhanced_ab_mcts_a import EnhancedABMCTSA
        from multi_llm_agent.ab_mcts import top_k
        from multi_llm_agent.ab_mcts.algorithm import LLMResponse
        
        agents = create_test_agents()
        
        def create_generate_fn(agent, name):
            def generate_fn(parent_state=None):
                result = agent.generate("test prompt")
                response = LLMResponse(
                    content=result["content"],
                    model_name=name,
                    score=result["quality"],
                    metadata=result
                )
                return response, response.score
            return generate_fn
        
        generate_fns = {name: create_generate_fn(agent, name) for name, agent in agents.items()}
        
        exploration_factors = [0.5, 1.0, 1.5, 2.0, 3.0]
        results = {}
        
        for factor in exploration_factors:
            # Reset agent call counts
            for agent in agents.values():
                agent.call_count = 0
            
            logger.info(f"Testing exploration factor: {factor}")
            
            algo = EnhancedABMCTSA(exploration_factor=factor)
            state = algo.init_tree()
            
            for i in range(20):
                state = algo.step(state, generate_fns)
            
            final_results = top_k(state, algo, k=3)
            
            results[factor] = {
                "best_score": final_results[0][1] if final_results else 0.0,
                "tree_size": len(state.tree),
                "total_calls": sum(agent.call_count for agent in agents.values())
            }
        
        # Analysis
        logger.info(f"\n📈 EXPLORATION FACTOR ANALYSIS:")
        logger.info("Factor\tBest Score\tTree Size\tAgent Calls")
        for factor, result in results.items():
            logger.info(f"{factor}\t{result['best_score']:.4f}\t\t{result['tree_size']}\t{result['total_calls']}")
        
        # Find optimal factor
        best_factor = max(results.items(), key=lambda x: x[1]['best_score'])
        logger.info(f"\n🎯 Best Exploration Factor: {best_factor[0]} (score: {best_factor[1]['best_score']:.4f})")
        
        return results
        
    except Exception as e:
        logger.error(f"Exploration factor test failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

def main():
    """Run all enhanced AB-MCTS tests."""
    
    logger.info("🧪 Enhanced AB-MCTS Test Suite")
    logger.info("=" * 70)
    
    try:
        # Comparison test
        comparison_results = run_algorithm_comparison(num_iterations=25)
        
        # Exploration factor test
        exploration_results = test_exploration_factors()
        
        logger.info("\n✅ All enhanced tests completed!")
        return 0
        
    except Exception as e:
        logger.error(f"Enhanced test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
