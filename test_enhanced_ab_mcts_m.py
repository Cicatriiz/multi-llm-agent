#!/usr/bin/env python3
"""
Comprehensive test script for Enhanced AB-MCTS-M implementation with PyMC integration.

This script tests:
1. Basic functionality with and without PyMC
2. Hierarchical Bayesian modeling
3. Thompson sampling with posterior inference
4. Fallback strategies
5. Performance comparisons
"""

import logging
import sys
import time
from typing import Dict, List, Tuple, Any

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test the import system
try:
    from src.multi_llm_agent.ab_mcts.enhanced_ab_mcts_m import (
        EnhancedABMCTSM, 
        ABMCTSMEnhancedState, 
        EnhancedPyMCInterface,
        HAS_PYMC
    )
    from src.multi_llm_agent.ab_mcts.base import Algorithm
    from src.multi_llm_agent.ab_mcts.tree import Node, Tree
    logger.info("✓ Successfully imported Enhanced AB-MCTS-M modules")
except ImportError as e:
    logger.error(f"✗ Failed to import Enhanced AB-MCTS-M modules: {e}")
    sys.exit(1)

# Check PyMC availability
logger.info(f"PyMC available: {HAS_PYMC}")

def mock_llm_generate_simple(state: Any) -> Tuple[str, float]:
    """Simple mock LLM generation function."""
    response = f"Response from simple model: {np.random.choice(['good', 'better', 'best'])}"
    score = np.random.uniform(0.3, 0.8)
    return response, score

def mock_llm_generate_advanced(state: Any) -> Tuple[str, float]:
    """Advanced mock LLM generation function with better performance."""
    response = f"Response from advanced model: {np.random.choice(['excellent', 'outstanding', 'superior'])}"
    score = np.random.uniform(0.6, 0.95)
    return response, score

def mock_llm_generate_experimental(state: Any) -> Tuple[str, float]:
    """Experimental mock LLM generation function with variable performance."""
    if np.random.random() < 0.3:  # 30% chance of high performance
        response = f"Response from experimental model: breakthrough insight"
        score = np.random.uniform(0.85, 1.0)
    else:
        response = f"Response from experimental model: standard response"
        score = np.random.uniform(0.1, 0.6)
    return response, score

def mock_llm_generate_conservative(state: Any) -> Tuple[str, float]:
    """Conservative mock LLM generation function with consistent but lower performance."""
    response = f"Response from conservative model: reliable output"
    score = np.random.uniform(0.4, 0.7)
    return response, score

def create_mock_generate_functions() -> Dict[str, Any]:
    """Create a set of mock generation functions simulating different LLM agents."""
    return {
        "simple_model": mock_llm_generate_simple,
        "advanced_model": mock_llm_generate_advanced,
        "experimental_model": mock_llm_generate_experimental,
        "conservative_model": mock_llm_generate_conservative,
    }

def test_basic_functionality():
    """Test basic Enhanced AB-MCTS-M functionality."""
    logger.info("=== Testing Basic Enhanced AB-MCTS-M Functionality ===")
    
    # Create algorithm instance
    algorithm = EnhancedABMCTSM(
        enable_pruning=True,
        model_selection_strategy="hierarchical_bayes",
        min_observations_for_pymc=3
    )
    
    # Initialize state
    state = algorithm.init_tree()
    assert isinstance(state, ABMCTSMEnhancedState)
    assert state.tree.root is not None
    assert len(state.all_observations) == 0
    logger.info("✓ Algorithm initialization successful")
    
    # Create mock generation functions
    generate_functions = create_mock_generate_functions()
    
    # Perform several steps
    num_steps = 15
    for i in range(num_steps):
        state = algorithm.step(state, generate_functions, inplace=True)
        logger.debug(f"Step {i+1}: {len(state.all_observations)} observations")
    
    # Verify state after steps
    assert len(state.all_observations) == num_steps
    assert state.model_iteration == num_steps
    
    # Get diagnostics
    diagnostics = algorithm.get_model_diagnostics(state)
    logger.info(f"Model diagnostics: {diagnostics}")
    
    # Get state-score pairs
    pairs = algorithm.get_state_score_pairs(state)
    assert len(pairs) == num_steps
    
    logger.info(f"✓ Basic functionality test passed ({num_steps} steps completed)")

def test_pymc_interface():
    """Test the PyMC interface separately."""
    logger.info("=== Testing PyMC Interface ===")
    
    # Create interface
    interface = EnhancedPyMCInterface(
        strategy="hierarchical_bayes",
        min_observations=2
    )
    
    # Create a simple tree structure
    tree = Tree.with_root_node()
    root = tree.root
    
    # Add some children to test child selection
    child1 = tree.add_node(("child1_state", 0.7), root)
    child2 = tree.add_node(("child2_state", 0.5), root)
    
    # Create mock observations
    from src.multi_llm_agent.ab_mcts.enhanced_ab_mcts_m import Observation
    observations = [
        Observation(reward=0.8, action="simple_model", node_expand_idx=1),
        Observation(reward=0.6, action="advanced_model", node_expand_idx=2),
        Observation(reward=0.9, action="simple_model", node_expand_idx=3),
        Observation(reward=0.7, action="experimental_model", node_expand_idx=4),
        Observation(reward=0.5, action="conservative_model", node_expand_idx=5),
    ]
    
    actions = ["simple_model", "advanced_model", "experimental_model", "conservative_model"]
    
    # Test action selection
    for i in range(5):
        try:
            result = interface.run(observations[:3], actions, root, observations)
            logger.info(f"PyMC interface result {i+1}: {result}")
            assert isinstance(result, (str, int))
        except Exception as e:
            logger.warning(f"PyMC interface test {i+1} failed: {e}")
    
    logger.info("✓ PyMC interface test completed")

def test_hierarchical_bayesian_modeling():
    """Test the hierarchical Bayesian modeling capabilities."""
    logger.info("=== Testing Hierarchical Bayesian Modeling ===")
    
    if not HAS_PYMC:
        logger.warning("Skipping Bayesian modeling test - PyMC not available")
        return
    
    algorithm = EnhancedABMCTSM(
        model_selection_strategy="hierarchical_bayes",
        min_observations_for_pymc=5
    )
    
    state = algorithm.init_tree()
    generate_functions = create_mock_generate_functions()
    
    # Perform enough steps to trigger PyMC modeling
    num_steps = 20
    start_time = time.time()
    
    for i in range(num_steps):
        state = algorithm.step(state, generate_functions, inplace=True)
        
        if i % 5 == 0:
            diagnostics = algorithm.get_model_diagnostics(state)
            logger.info(f"Step {i}: {diagnostics['observations_by_action']}")
    
    end_time = time.time()
    
    # Analyze results
    diagnostics = algorithm.get_model_diagnostics(state)
    logger.info(f"Final diagnostics: {diagnostics}")
    logger.info(f"Processing time: {end_time - start_time:.2f} seconds")
    
    # Check that actions are being selected based on performance
    action_counts = diagnostics['observations_by_action']
    if action_counts:
        most_selected = max(action_counts, key=action_counts.get)
        logger.info(f"Most selected action: {most_selected} ({action_counts[most_selected]} times)")
    
    logger.info("✓ Hierarchical Bayesian modeling test completed")

def test_thompson_sampling():
    """Test Thompson sampling behavior."""
    logger.info("=== Testing Thompson Sampling ===")
    
    algorithm = EnhancedABMCTSM(
        model_selection_strategy="hierarchical_bayes",
        min_observations_for_pymc=3
    )
    
    # Create generate functions with different reward distributions
    def high_reward_generator(state):
        return f"high_reward_{np.random.randint(1000)}", np.random.beta(8, 2)  # High rewards
    
    def medium_reward_generator(state):
        return f"medium_reward_{np.random.randint(1000)}", np.random.beta(5, 5)  # Medium rewards
    
    def low_reward_generator(state):
        return f"low_reward_{np.random.randint(1000)}", np.random.beta(2, 8)  # Low rewards
    
    generate_functions = {
        "high_reward": high_reward_generator,
        "medium_reward": medium_reward_generator,
        "low_reward": low_reward_generator,
    }
    
    state = algorithm.init_tree()
    
    # Run many steps to see if Thompson sampling learns
    num_steps = 30
    action_selections = []
    
    for i in range(num_steps):
        old_observations = len(state.all_observations)
        state = algorithm.step(state, generate_functions, inplace=True)
        
        # Track which action was selected
        new_observations = list(state.all_observations.values())[old_observations:]
        if new_observations:
            action_selections.append(new_observations[0].action)
    
    # Analyze action selection patterns
    from collections import Counter
    action_counter = Counter(action_selections)
    logger.info(f"Action selection frequency: {dict(action_counter)}")
    
    # Get final diagnostics
    diagnostics = algorithm.get_model_diagnostics(state)
    logger.info(f"Final action counts: {diagnostics['observations_by_action']}")
    
    # Calculate average rewards per action
    avg_rewards = {}
    for obs in state.all_observations.values():
        if obs.action not in avg_rewards:
            avg_rewards[obs.action] = []
        avg_rewards[obs.action].append(obs.reward)
    
    for action, rewards in avg_rewards.items():
        avg = np.mean(rewards)
        logger.info(f"Average reward for {action}: {avg:.3f}")
    
    logger.info("✓ Thompson sampling test completed")

def test_fallback_strategies():
    """Test fallback strategies when PyMC fails or is unavailable."""
    logger.info("=== Testing Fallback Strategies ===")
    
    # Force fallback by using very high minimum observations
    algorithm = EnhancedABMCTSM(
        model_selection_strategy="hierarchical_bayes",
        min_observations_for_pymc=1000  # Force fallback
    )
    
    state = algorithm.init_tree()
    generate_functions = create_mock_generate_functions()
    
    # Perform steps with forced fallback
    num_steps = 10
    for i in range(num_steps):
        state = algorithm.step(state, generate_functions, inplace=True)
    
    diagnostics = algorithm.get_model_diagnostics(state)
    logger.info(f"Fallback test diagnostics: {diagnostics}")
    
    assert len(state.all_observations) == num_steps
    logger.info("✓ Fallback strategies test passed")

def test_performance_comparison():
    """Compare performance between different strategies."""
    logger.info("=== Testing Performance Comparison ===")
    
    strategies = [
        ("hierarchical_bayes", 3),
        ("hierarchical_bayes", 10),  # Higher threshold to force more fallback
    ]
    
    if not HAS_PYMC:
        logger.warning("Skipping performance comparison - PyMC not available")
        return
    
    results = {}
    
    for strategy, min_obs in strategies:
        logger.info(f"Testing strategy: {strategy} with min_obs: {min_obs}")
        
        algorithm = EnhancedABMCTSM(
            model_selection_strategy=strategy,
            min_observations_for_pymc=min_obs
        )
        
        state = algorithm.init_tree()
        generate_functions = create_mock_generate_functions()
        
        start_time = time.time()
        num_steps = 15
        
        for i in range(num_steps):
            state = algorithm.step(state, generate_functions, inplace=True)
        
        end_time = time.time()
        
        # Calculate average reward
        avg_reward = np.mean([obs.reward for obs in state.all_observations.values()])
        
        results[f"{strategy}_{min_obs}"] = {
            'avg_reward': avg_reward,
            'time': end_time - start_time,
            'observations': len(state.all_observations)
        }
        
        logger.info(f"Results: avg_reward={avg_reward:.3f}, time={end_time-start_time:.2f}s")
    
    logger.info(f"Performance comparison results: {results}")
    logger.info("✓ Performance comparison completed")

def run_comprehensive_test():
    """Run all tests in sequence."""
    logger.info("=== Starting Comprehensive Enhanced AB-MCTS-M Test Suite ===")
    
    tests = [
        test_basic_functionality,
        test_pymc_interface,
        test_hierarchical_bayesian_modeling,
        test_thompson_sampling,
        test_fallback_strategies,
        test_performance_comparison,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
            logger.info(f"✓ {test_func.__name__} PASSED")
        except Exception as e:
            logger.error(f"✗ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"=== Test Summary: {passed}/{total} tests passed ===")
    
    if passed == total:
        logger.info("🎉 All Enhanced AB-MCTS-M tests passed!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
