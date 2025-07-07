"""
Adaptive Model Selector with AB-MCTS inspired intelligent switching.

This module implements sophisticated model selection strategies that learn
from performance data and adapt to different contexts and tasks.
"""

import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .base import (
    ModelInterface, ModelSelector, GenerationRequest, GenerationResult, 
    TaskContext, ModelType, ConfigManager
)


@dataclass
class ModelPerformanceData:
    """Performance tracking data for a model."""
    total_requests: int = 0
    total_cost: float = 0.0
    total_response_time: float = 0.0
    success_count: int = 0
    quality_scores: List[float] = field(default_factory=list)
    context_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    recent_performance: deque = field(default_factory=lambda: deque(maxlen=50))
    last_updated: float = field(default_factory=time.time)


@dataclass
class MCTSNode:
    """Node in the MCTS tree for model selection."""
    model_name: str
    context: str
    visits: int = 0
    total_reward: float = 0.0
    children: Dict[str, 'MCTSNode'] = field(default_factory=dict)
    parent: Optional['MCTSNode'] = None
    
    @property
    def average_reward(self) -> float:
        """Average reward for this node."""
        return self.total_reward / max(self.visits, 1)
    
    def ucb_score(self, exploration_constant: float = 1.4) -> float:
        """Calculate UCB (Upper Confidence Bound) score."""
        if self.visits == 0:
            return float('inf')
        
        if self.parent and self.parent.visits > 0:
            exploration = exploration_constant * math.sqrt(
                math.log(self.parent.visits) / self.visits
            )
        else:
            exploration = 0
        
        return self.average_reward + exploration


class AdaptiveModelSelector(ModelSelector):
    """
    Adaptive model selector using AB-MCTS inspired approach.
    
    This selector learns from past performance and intelligently chooses
    the best model for each request based on context, performance history,
    and exploration needs.
    """
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.performance_data: Dict[str, ModelPerformanceData] = {}
        self.mcts_tree: Dict[str, MCTSNode] = {}
        self.context_preferences = config.get("selection.context_preferences", {})
        
        # AB-MCTS parameters
        self.exploration_constant = config.get("ab_mcts.exploration_constant", 1.4)
        self.selection_temperature = config.get("ab_mcts.selection_temperature", 0.1)
        self.max_iterations = config.get("ab_mcts.max_iterations", 25)
        
        # Adaptive parameters
        self.learning_rate = 0.1
        self.decay_factor = 0.95
        self.min_exploration_rate = 0.1
        self.exploration_rate = 0.3
        
        # Performance weights
        self.weights = {
            "response_time": -0.3,  # Negative because lower is better
            "cost": -0.2,          # Negative because lower is better
            "quality": 0.4,        # Positive because higher is better
            "success_rate": 0.3,   # Positive because higher is better
            "context_match": 0.2   # Positive for context appropriateness
        }
    
    def select_model(self, 
                    request: GenerationRequest, 
                    available_models: List[ModelInterface]) -> ModelInterface:
        """
        Select the best model using adaptive AB-MCTS approach.
        """
        if not available_models:
            raise ValueError("No available models")
        
        if len(available_models) == 1:
            return available_models[0]
        
        # Get context-specific preferences
        context_prefs = self._get_context_preferences(request.context)
        
        # If we have a specific model preference, try to use it
        if request.model_preference:
            preferred_model = next(
                (m for m in available_models if m.config.name == request.model_preference),
                None
            )
            if preferred_model:
                return preferred_model
        
        # Use MCTS-inspired selection for complex scenarios
        if len(available_models) > 2 and self._should_use_mcts(request):
            return self._mcts_select(request, available_models)
        
        # Use adaptive scoring for simpler scenarios
        return self._adaptive_select(request, available_models, context_prefs)
    
    def _should_use_mcts(self, request: GenerationRequest) -> bool:
        """Determine if MCTS selection should be used."""
        # Use MCTS for complex reasoning tasks or when we have sufficient data
        complex_contexts = {TaskContext.REASONING, TaskContext.ANALYSIS, TaskContext.PREMIUM_TASKS}
        has_history = len(self.performance_data) > 3
        
        return request.context in complex_contexts or has_history
    
    def _mcts_select(self, 
                    request: GenerationRequest, 
                    available_models: List[ModelInterface]) -> ModelInterface:
        """Select model using MCTS-inspired approach."""
        context_key = request.context.value
        
        # Initialize tree nodes if needed
        if context_key not in self.mcts_tree:
            self.mcts_tree[context_key] = MCTSNode("root", context_key)
        
        root = self.mcts_tree[context_key]
        
        # Ensure all available models have nodes
        for model in available_models:
            if model.config.name not in root.children:
                root.children[model.config.name] = MCTSNode(
                    model.config.name, context_key, parent=root
                )
        
        # Run MCTS iterations (simplified)
        for _ in range(min(self.max_iterations, 10)):  # Limit for performance
            self._mcts_iteration(root, available_models)
        
        # Select best model based on visit count and average reward
        best_model_name = self._select_best_from_mcts(root, available_models)
        
        return next(m for m in available_models if m.config.name == best_model_name)
    
    def _mcts_iteration(self, root: MCTSNode, available_models: List[ModelInterface]):
        """Perform one MCTS iteration."""
        # Selection: choose best child based on UCB
        node = root
        path = [node]
        
        while node.children and all(child.visits > 0 for child in node.children.values()):
            available_children = [
                child for child in node.children.values()
                if any(m.config.name == child.model_name for m in available_models)
            ]
            
            if not available_children:
                break
                
            node = max(available_children, key=lambda x: x.ucb_score(self.exploration_constant))
            path.append(node)
        
        # Expansion: add unvisited children
        if node.children:
            unvisited = [
                child for child in node.children.values()
                if child.visits == 0 and any(m.config.name == child.model_name for m in available_models)
            ]
            if unvisited:
                node = random.choice(unvisited)
                path.append(node)
        
        # Simulation: estimate performance
        reward = self._simulate_performance(node.model_name, node.context)
        
        # Backpropagation: update all nodes in path
        for node in reversed(path):
            node.visits += 1
            node.total_reward += reward
    
    def _simulate_performance(self, model_name: str, context: str) -> float:
        """Simulate performance for a model in a given context."""
        if model_name in self.performance_data:
            perf_data = self.performance_data[model_name]
            
            # Use historical performance if available
            if context in perf_data.context_performance:
                context_perf = perf_data.context_performance[context]
                return context_perf.get("composite_score", 0.5)
            
            # Use overall performance as fallback
            if perf_data.quality_scores:
                avg_quality = sum(perf_data.quality_scores) / len(perf_data.quality_scores)
                success_rate = perf_data.success_count / max(perf_data.total_requests, 1)
                return (avg_quality + success_rate) / 2
        
        # Default neutral performance for unknown models
        return 0.5 + random.uniform(-0.1, 0.1)
    
    def _select_best_from_mcts(self, 
                             root: MCTSNode, 
                             available_models: List[ModelInterface]) -> str:
        """Select the best model from MCTS tree."""
        available_children = [
            child for child in root.children.values()
            if any(m.config.name == child.model_name for m in available_models)
        ]
        
        if not available_children:
            return available_models[0].config.name
        
        # Use softmax selection with temperature
        scores = [child.average_reward for child in available_children]
        if self.selection_temperature > 0:
            # Apply temperature for exploration
            exp_scores = [math.exp(score / self.selection_temperature) for score in scores]
            total = sum(exp_scores)
            probabilities = [score / total for score in exp_scores]
            
            # Sample based on probabilities
            rand_val = random.random()
            cumulative = 0
            for i, prob in enumerate(probabilities):
                cumulative += prob
                if rand_val <= cumulative:
                    return available_children[i].model_name
        
        # Fallback to best scoring
        best_child = max(available_children, key=lambda x: x.average_reward)
        return best_child.model_name
    
    def _adaptive_select(self, 
                        request: GenerationRequest,
                        available_models: List[ModelInterface],
                        context_prefs: Dict[str, Any]) -> ModelInterface:
        """Select model using adaptive scoring."""
        model_scores = {}
        
        for model in available_models:
            score = self._calculate_model_score(model, request, context_prefs)
            model_scores[model.config.name] = score
        
        # Add exploration bonus
        for model_name in model_scores:
            if model_name not in self.performance_data or \
               self.performance_data[model_name].total_requests < 5:
                model_scores[model_name] += self.exploration_rate
        
        # Select best model
        best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x])
        return next(m for m in available_models if m.config.name == best_model_name)
    
    def _calculate_model_score(self, 
                             model: ModelInterface,
                             request: GenerationRequest,
                             context_prefs: Dict[str, Any]) -> float:
        """Calculate comprehensive score for a model."""
        score = 0.0
        model_name = model.config.name
        
        # Context appropriateness
        context_score = self._get_context_score(model, request.context, context_prefs)
        score += self.weights["context_match"] * context_score
        
        # Historical performance
        if model_name in self.performance_data:
            perf_data = self.performance_data[model_name]
            
            # Response time (normalized, lower is better)
            if perf_data.total_requests > 0:
                avg_response_time = perf_data.total_response_time / perf_data.total_requests
                time_score = max(0, 1 - (avg_response_time / 30))  # Normalize to 30s max
                score += self.weights["response_time"] * time_score
            
            # Cost (normalized, lower is better)
            if perf_data.total_requests > 0:
                avg_cost = perf_data.total_cost / perf_data.total_requests
                cost_score = max(0, 1 - (avg_cost / 0.1))  # Normalize to $0.1 max
                score += self.weights["cost"] * cost_score
            
            # Quality
            if perf_data.quality_scores:
                avg_quality = sum(perf_data.quality_scores) / len(perf_data.quality_scores)
                score += self.weights["quality"] * avg_quality
            
            # Success rate
            success_rate = perf_data.success_count / max(perf_data.total_requests, 1)
            score += self.weights["success_rate"] * success_rate
            
            # Recent performance trend
            if perf_data.recent_performance:
                recent_trend = self._calculate_trend(perf_data.recent_performance)
                score += 0.1 * recent_trend  # Small bonus for improving models
        
        # Model capabilities bonus
        capability_bonus = self._get_capability_bonus(model, request)
        score += capability_bonus
        
        return score
    
    def _get_context_score(self, 
                          model: ModelInterface,
                          context: TaskContext,
                          context_prefs: Dict[str, Any]) -> float:
        """Get context appropriateness score."""
        # Check if model is in preferred list for this context
        context_key = context.value
        if context_key in context_prefs:
            prefs = context_prefs[context_key]
            if "primary" in prefs and model.config.name in prefs["primary"]:
                return 1.0
            elif "fallback" in prefs and model.config.name in prefs["fallback"]:
                return 0.7
        
        # Check model's native context support
        if context in model.config.contexts:
            return 0.8
        
        # Check model type compatibility
        context_model_type_map = {
            TaskContext.CODING: [ModelType.CODING, ModelType.GENERAL],
            TaskContext.REASONING: [ModelType.REASONING, ModelType.GENERAL],
            TaskContext.ANALYSIS: [ModelType.REASONING, ModelType.GENERAL],
            TaskContext.CHAT: [ModelType.FAST, ModelType.GENERAL],
            TaskContext.CREATIVE: [ModelType.GENERAL, ModelType.MULTIMODAL],
            TaskContext.MULTIMODAL: [ModelType.MULTIMODAL, ModelType.GENERAL],
        }
        
        compatible_types = context_model_type_map.get(context, [ModelType.GENERAL])
        if model.config.model_type in compatible_types:
            return 0.6
        
        return 0.3  # Default lower score for incompatible contexts
    
    def _get_capability_bonus(self, 
                            model: ModelInterface,
                            request: GenerationRequest) -> float:
        """Get bonus score based on model capabilities."""
        bonus = 0.0
        
        # Tool usage bonus
        if request.tools and "function_calling" in [cap.value for cap in model.config.capabilities]:
            bonus += 0.1
        
        # Streaming bonus
        if request.stream and "streaming" in [cap.value for cap in model.config.capabilities]:
            bonus += 0.05
        
        return bonus
    
    def _calculate_trend(self, recent_data: deque) -> float:
        """Calculate performance trend from recent data."""
        if len(recent_data) < 3:
            return 0.0
        
        # Simple linear trend calculation
        data_points = list(recent_data)
        x = np.arange(len(data_points))
        y = np.array(data_points)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        return np.tanh(slope)  # Normalize to [-1, 1]
    
    def _get_context_preferences(self, context: TaskContext) -> Dict[str, Any]:
        """Get context-specific preferences."""
        return self.context_preferences.get(context.value, {})
    
    def update_performance(self, model: ModelInterface, result: GenerationResult):
        """Update performance data based on generation result."""
        model_name = model.config.name
        
        # Initialize performance data if needed
        if model_name not in self.performance_data:
            self.performance_data[model_name] = ModelPerformanceData()
        
        perf_data = self.performance_data[model_name]
        
        # Update overall metrics
        perf_data.total_requests += 1
        perf_data.total_cost += result.cost
        perf_data.total_response_time += result.response_time
        
        # Determine success (simple heuristic)
        success = result.response_time < 60 and len(result.message.content) > 10
        if success:
            perf_data.success_count += 1
        
        # Update quality scores
        if result.quality_score is not None:
            perf_data.quality_scores.append(result.quality_score)
            # Keep only recent quality scores
            if len(perf_data.quality_scores) > 100:
                perf_data.quality_scores = perf_data.quality_scores[-100:]
        
        # Update context-specific performance
        context_key = "unknown"
        for msg in result.message.metadata.get("request_context", []):
            if hasattr(msg, "context"):
                context_key = msg.context.value
                break
        
        if context_key not in perf_data.context_performance:
            perf_data.context_performance[context_key] = {
                "requests": 0,
                "success_rate": 0.0,
                "avg_quality": 0.0,
                "avg_cost": 0.0,
                "avg_response_time": 0.0,
                "composite_score": 0.5
            }
        
        context_perf = perf_data.context_performance[context_key]
        context_perf["requests"] += 1
        
        # Update context metrics with exponential moving average
        alpha = 0.1  # Learning rate
        context_perf["avg_cost"] = (1 - alpha) * context_perf["avg_cost"] + alpha * result.cost
        context_perf["avg_response_time"] = (1 - alpha) * context_perf["avg_response_time"] + alpha * result.response_time
        
        if result.quality_score is not None:
            context_perf["avg_quality"] = (1 - alpha) * context_perf["avg_quality"] + alpha * result.quality_score
        
        success_rate = perf_data.success_count / perf_data.total_requests
        context_perf["success_rate"] = success_rate
        
        # Calculate composite score
        normalized_cost = max(0, 1 - (context_perf["avg_cost"] / 0.1))
        normalized_time = max(0, 1 - (context_perf["avg_response_time"] / 30))
        
        context_perf["composite_score"] = (
            0.3 * context_perf["avg_quality"] +
            0.3 * success_rate +
            0.2 * normalized_time +
            0.2 * normalized_cost
        )
        
        # Update recent performance
        perf_data.recent_performance.append(context_perf["composite_score"])
        perf_data.last_updated = time.time()
        
        # Decay exploration rate over time
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.decay_factor
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all models."""
        summary = {
            "total_models": len(self.performance_data),
            "exploration_rate": self.exploration_rate,
            "models": {}
        }
        
        for model_name, perf_data in self.performance_data.items():
            if perf_data.total_requests > 0:
                avg_quality = (
                    sum(perf_data.quality_scores) / len(perf_data.quality_scores)
                    if perf_data.quality_scores else 0.0
                )
                
                summary["models"][model_name] = {
                    "total_requests": perf_data.total_requests,
                    "success_rate": perf_data.success_count / perf_data.total_requests,
                    "avg_cost": perf_data.total_cost / perf_data.total_requests,
                    "avg_response_time": perf_data.total_response_time / perf_data.total_requests,
                    "avg_quality": avg_quality,
                    "contexts": list(perf_data.context_performance.keys())
                }
        
        return summary
