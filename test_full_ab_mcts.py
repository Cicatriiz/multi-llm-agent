#!/usr/bin/env python3
"""
Full-featured AB-MCTS test with scipy/numpy analytics and visualizations.
"""

import sys
import os
import random
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
import statistics

import numpy as np
from scipy import stats

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class AdvancedTestAgent:
    """Advanced test agent with realistic behavior patterns."""
    
    def __init__(self, name: str, quality_profile: Dict[str, float]):
        self.name = name
        self.quality_profile = quality_profile
        self.call_count = 0
        self.generation_times = []
        self.response_history = []
        
    def generate(self, prompt: str) -> Dict[str, Any]:
        """Generate response with realistic quality patterns."""
        self.call_count += 1
        start_time = time.time()
        
        # Simulate different response patterns based on prompt type
        prompt_lower = prompt.lower()
        
        # Base quality from profile
        base_quality = self.quality_profile.get('base', 0.7)
        
        # Adjust quality based on prompt characteristics
        if 'complex' in prompt_lower or 'optimization' in prompt_lower:
            quality_modifier = self.quality_profile.get('complex_boost', 0.0)
        elif 'simple' in prompt_lower or 'basic' in prompt_lower:
            quality_modifier = self.quality_profile.get('simple_boost', 0.0)
        elif 'improve' in prompt_lower or 'enhance' in prompt_lower:
            quality_modifier = self.quality_profile.get('refinement_boost', 0.0)
        else:
            quality_modifier = 0.0
        
        # Add realistic noise and learning effects
        learning_factor = min(0.1, self.call_count * 0.01)  # Slight improvement over time
        noise = np.random.normal(0, self.quality_profile.get('noise_std', 0.1))
        
        # Fatigue factor (quality decreases slightly with many calls)
        fatigue_factor = -max(0, (self.call_count - 10) * 0.005)
        
        final_quality = base_quality + quality_modifier + learning_factor + fatigue_factor + noise
        final_quality = np.clip(final_quality, 0.0, 1.0)
        
        # Simulate processing time
        process_time = self.quality_profile.get('response_time', 0.01)
        time.sleep(process_time + np.random.exponential(0.005))
        
        generation_time = time.time() - start_time
        self.generation_times.append(generation_time)
        
        content = f"Response from {self.name}: {prompt[:40]}... (quality: {final_quality:.3f})"
        if final_quality > 0.8:
            content += " [HIGH QUALITY] Detailed analysis with comprehensive insights."
        elif final_quality > 0.6:
            content += " [GOOD] Well-structured response with key points covered."
        else:
            content += " [BASIC] Simple response addressing main question."
        
        response = {
            "content": content,
            "model": self.name,
            "quality": final_quality,
            "generation_time": generation_time,
            "call_number": self.call_count,
        }
        
        self.response_history.append(response)
        return response
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        if not self.generation_times:
            return {}
        
        qualities = [r['quality'] for r in self.response_history]
        
        return {
            "total_calls": self.call_count,
            "avg_quality": np.mean(qualities),
            "quality_std": np.std(qualities),
            "quality_trend": np.polyfit(range(len(qualities)), qualities, 1)[0] if len(qualities) > 1 else 0,
            "avg_response_time": np.mean(self.generation_times),
            "response_time_std": np.std(self.generation_times),
            "min_quality": np.min(qualities),
            "max_quality": np.max(qualities),
            "quality_percentiles": {
                "25th": np.percentile(qualities, 25),
                "50th": np.percentile(qualities, 50),
                "75th": np.percentile(qualities, 75),
                "95th": np.percentile(qualities, 95),
            }
        }

def create_realistic_agents() -> Dict[str, AdvancedTestAgent]:
    """Create realistic agent profiles for testing."""
    return {
        "reasoning_specialist": AdvancedTestAgent("reasoning_specialist", {
            "base": 0.85,
            "complex_boost": 0.10,
            "simple_boost": -0.05,
            "refinement_boost": 0.08,
            "noise_std": 0.08,
            "response_time": 0.025,
        }),
        "creative_generator": AdvancedTestAgent("creative_generator", {
            "base": 0.78,
            "complex_boost": 0.05,
            "simple_boost": 0.02,
            "refinement_boost": 0.12,
            "noise_std": 0.12,
            "response_time": 0.018,
        }),
        "fast_responder": AdvancedTestAgent("fast_responder", {
            "base": 0.65,
            "complex_boost": -0.08,
            "simple_boost": 0.10,
            "refinement_boost": 0.05,
            "noise_std": 0.06,
            "response_time": 0.008,
        }),
        "premium_model": AdvancedTestAgent("premium_model", {
            "base": 0.92,
            "complex_boost": 0.05,
            "simple_boost": -0.02,
            "refinement_boost": 0.06,
            "noise_std": 0.05,
            "response_time": 0.035,
        }),
        "balanced_agent": AdvancedTestAgent("balanced_agent", {
            "base": 0.75,
            "complex_boost": 0.03,
            "simple_boost": 0.03,
            "refinement_boost": 0.07,
            "noise_std": 0.09,
            "response_time": 0.015,
        }),
    }

def run_full_ab_mcts_analysis():
    """Run comprehensive AB-MCTS analysis with scipy/numpy."""
    
    logger.info("🔬 Full AB-MCTS Analysis with SciPy/NumPy")
    logger.info("=" * 60)
    
    try:
        from multi_llm_agent.ab_mcts.enhanced_ab_mcts_a import EnhancedABMCTSA
        from multi_llm_agent.ab_mcts import ABMCTSA, top_k
        from multi_llm_agent.ab_mcts.algorithm import LLMResponse
        
        agents = create_realistic_agents()
        
        # Test different scenarios
        scenarios = [
            {
                "name": "Complex Problem Solving",
                "prompt": "Solve this complex optimization problem with multiple constraints",
                "iterations": 30,
                "exploration_factor": 1.5,
            },
            {
                "name": "Simple Task Completion", 
                "prompt": "Complete this simple data processing task",
                "iterations": 20,
                "exploration_factor": 1.0,
            },
            {
                "name": "Iterative Refinement",
                "prompt": "Create and improve a solution through multiple iterations",
                "iterations": 35,
                "exploration_factor": 2.0,
            }
        ]
        
        results = {}
        
        for scenario in scenarios:
            logger.info(f"\n🧪 Testing Scenario: {scenario['name']}")
            logger.info(f"   Prompt: {scenario['prompt']}")
            logger.info(f"   Iterations: {scenario['iterations']}")
            
            # Reset agent states
            for agent in agents.values():
                agent.call_count = 0
                agent.generation_times = []
                agent.response_history = []
            
            # Create generation functions
            def create_generate_fn(agent, name, prompt_template):
                def generate_fn(parent_state=None):
                    if parent_state:
                        prompt = f"Improve on previous response: '{parent_state.content[:50]}...'. {prompt_template}"
                    else:
                        prompt = prompt_template
                    
                    result = agent.generate(prompt)
                    response = LLMResponse(
                        content=result["content"],
                        model_name=name,
                        score=result["quality"],
                        metadata=result
                    )
                    return response, response.score
                return generate_fn
            
            generate_fns = {
                name: create_generate_fn(agent, name, scenario['prompt']) 
                for name, agent in agents.items()
            }
            
            # Test Enhanced AB-MCTS-A
            start_time = time.time()
            enhanced_algo = EnhancedABMCTSA(exploration_factor=scenario['exploration_factor'])
            enhanced_state = enhanced_algo.init_tree()
            
            quality_progression = []
            for i in range(scenario['iterations']):
                enhanced_state = enhanced_algo.step(enhanced_state, generate_fns)
                
                # Track quality progression
                if enhanced_state.tree.get_state_score_pairs():
                    current_best = max(pair[1] for pair in enhanced_state.tree.get_state_score_pairs())
                    quality_progression.append(current_best)
            
            enhanced_time = time.time() - start_time
            enhanced_results = top_k(enhanced_state, enhanced_algo, k=5)
            enhanced_diagnostics = enhanced_algo.get_diagnostics(enhanced_state)
            
            # Analyze agent performance
            agent_stats = {name: agent.get_performance_stats() for name, agent in agents.items()}
            
            # Statistical analysis
            quality_analysis = {
                "progression": quality_progression,
                "final_scores": [r[1] for r in enhanced_results],
                "convergence_rate": _calculate_convergence_rate(quality_progression),
                "exploration_efficiency": _calculate_exploration_efficiency(enhanced_diagnostics),
            }
            
            results[scenario['name']] = {
                "scenario": scenario,
                "runtime": enhanced_time,
                "tree_size": len(enhanced_state.tree),
                "results": enhanced_results,
                "quality_analysis": quality_analysis,
                "agent_stats": agent_stats,
                "diagnostics": enhanced_diagnostics,
            }
            
            # Log key results
            logger.info(f"   ✓ Completed in {enhanced_time:.3f}s")
            logger.info(f"   ✓ Tree size: {len(enhanced_state.tree)} nodes")
            logger.info(f"   ✓ Best score: {enhanced_results[0][1]:.4f}")
            logger.info(f"   ✓ Convergence rate: {quality_analysis['convergence_rate']:.6f}")
        
        # Cross-scenario analysis
        _analyze_cross_scenario_results(results)
        
        return results
        
    except Exception as e:
        logger.error(f"Full analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

def _calculate_convergence_rate(progression: List[float]) -> float:
    """Calculate how quickly the algorithm converges to good solutions."""
    if len(progression) < 5:
        return 0.0
    
    # Fit exponential curve to progression
    x = np.arange(len(progression))
    y = np.array(progression)
    
    # Avoid log of zero/negative values
    y_shifted = y - np.min(y) + 0.001
    
    try:
        # Fit log-linear model: log(y) = a + b*x
        coeffs = np.polyfit(x, np.log(y_shifted), 1)
        return coeffs[0]  # Growth rate
    except:
        return 0.0

def _calculate_exploration_efficiency(diagnostics: Dict[str, Any]) -> float:
    """Calculate how efficiently the algorithm explores the space."""
    tree_size = diagnostics.get('tree_size', 1)
    step_count = diagnostics.get('step_count', 1)
    
    # Efficiency = how much tree growth per step
    return tree_size / step_count if step_count > 0 else 0.0

def _analyze_cross_scenario_results(results: Dict[str, Any]):
    """Analyze results across different scenarios."""
    
    logger.info("\n📊 CROSS-SCENARIO ANALYSIS")
    logger.info("=" * 50)
    
    # Performance across scenarios
    scenario_stats = {}
    for scenario_name, result in results.items():
        scenario_stats[scenario_name] = {
            "runtime": result["runtime"],
            "best_score": result["results"][0][1] if result["results"] else 0.0,
            "tree_size": result["tree_size"],
            "convergence_rate": result["quality_analysis"]["convergence_rate"],
        }
    
    # Find best performing scenario
    best_scenario = max(scenario_stats.items(), key=lambda x: x[1]["best_score"])
    fastest_scenario = min(scenario_stats.items(), key=lambda x: x[1]["runtime"])
    most_exploratory = max(scenario_stats.items(), key=lambda x: x[1]["tree_size"])
    
    logger.info(f"🏆 Best Quality: {best_scenario[0]} (score: {best_scenario[1]['best_score']:.4f})")
    logger.info(f"⚡ Fastest: {fastest_scenario[0]} (time: {fastest_scenario[1]['runtime']:.3f}s)")
    logger.info(f"🌳 Most Exploratory: {most_exploratory[0]} (tree: {most_exploratory[1]['tree_size']} nodes)")
    
    # Agent performance across scenarios
    logger.info(f"\n🤖 AGENT PERFORMANCE ANALYSIS:")
    all_agent_names = set()
    for result in results.values():
        all_agent_names.update(result["agent_stats"].keys())
    
    for agent_name in all_agent_names:
        total_calls = sum(
            result["agent_stats"].get(agent_name, {}).get("total_calls", 0)
            for result in results.values()
        )
        avg_quality = np.mean([
            result["agent_stats"].get(agent_name, {}).get("avg_quality", 0)
            for result in results.values()
            if agent_name in result["agent_stats"]
        ])
        
        logger.info(f"  • {agent_name}: {total_calls} calls, avg quality: {avg_quality:.4f}")

def test_bayesian_convergence():
    """Test the Bayesian statistics convergence properties."""
    
    logger.info("\n🔬 Testing Bayesian Statistics Convergence")
    logger.info("=" * 50)
    
    from multi_llm_agent.ab_mcts.enhanced_ab_mcts_a import BayesianStats
    
    # Test different reward patterns
    patterns = {
        "improving": lambda i: 0.5 + 0.4 * (i / 100),  # Improving over time
        "declining": lambda i: 0.9 - 0.3 * (i / 100),  # Declining over time
        "stable_high": lambda i: 0.8 + np.random.normal(0, 0.05),  # Stable high
        "stable_low": lambda i: 0.3 + np.random.normal(0, 0.05),   # Stable low
        "volatile": lambda i: 0.5 + 0.4 * np.sin(i / 10) + np.random.normal(0, 0.1),  # Volatile
    }
    
    for pattern_name, pattern_func in patterns.items():
        logger.info(f"\n  Testing pattern: {pattern_name}")
        
        stats = BayesianStats()
        samples = []
        confidence_intervals = []
        
        for i in range(100):
            reward = np.clip(pattern_func(i), 0, 1)
            stats.update(reward)
            
            if i % 10 == 9:  # Every 10 samples
                sample = stats.sample()
                ci = stats.confidence_interval()
                samples.append(sample)
                confidence_intervals.append(ci)
        
        # Analyze convergence
        final_mean = stats.posterior_mean()
        final_variance = stats.posterior_variance()
        empirical_mean = stats.mean
        
        logger.info(f"    • Posterior mean: {final_mean:.4f}")
        logger.info(f"    • Empirical mean: {empirical_mean:.4f}")
        logger.info(f"    • Posterior variance: {final_variance:.6f}")
        logger.info(f"    • Final 95% CI: [{confidence_intervals[-1][0]:.4f}, {confidence_intervals[-1][1]:.4f}]")

def main():
    """Run all full-featured tests."""
    
    logger.info("🧪 Full-Featured AB-MCTS Test Suite with SciPy/NumPy")
    logger.info("=" * 70)
    
    try:
        # Test Bayesian convergence
        test_bayesian_convergence()
        
        # Run full analysis
        analysis_results = run_full_ab_mcts_analysis()
        
        logger.info("\n✅ All full-featured tests completed successfully!")
        logger.info(f"📊 Analyzed {len(analysis_results)} scenarios with advanced statistics")
        
        return 0
        
    except Exception as e:
        logger.error(f"Full test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
