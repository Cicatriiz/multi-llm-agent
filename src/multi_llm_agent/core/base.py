"""
Core base classes and interfaces for the Multi-LLM Agent Framework.

This module defines the fundamental abstractions that enable intelligent
model switching and agentic behavior.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
import uuid

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """Types of models available in the framework."""
    REASONING = "reasoning"
    CODING = "coding"
    GENERAL = "general"
    FAST = "fast"
    MULTIMODAL = "multimodal"


class TaskContext(str, Enum):
    """Task contexts that influence model selection."""
    CODING = "coding"
    ANALYSIS = "analysis"
    REASONING = "reasoning"
    CHAT = "chat"
    CREATIVE = "creative"
    MULTIMODAL = "multimodal"
    FAST_TASKS = "fast_tasks"
    PREMIUM_TASKS = "premium_tasks"


class AgentCapability(str, Enum):
    """Capabilities that agents can possess."""
    TEXT = "text"
    CODE = "code"
    REASONING = "reasoning"
    VISION = "vision"
    FUNCTION_CALLING = "function_calling"
    MULTIMODAL = "multimodal"
    WEB_SEARCH = "web_search"
    FILE_OPERATIONS = "file_operations"
    DATA_ANALYSIS = "data_analysis"


@dataclass
class ModelConfig:
    """Configuration for a language model."""
    name: str
    api_name: str
    provider: str  # openai, google, deepseek, etc.
    model_type: ModelType
    capabilities: List[AgentCapability]
    contexts: List[TaskContext]
    
    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # Cost information
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    
    # Performance characteristics
    avg_response_time: float = 5.0
    max_context_length: int = 128000
    
    # Availability
    is_available: bool = True
    error_rate: float = 0.0


@dataclass
class Message:
    """A message in a conversation."""
    role: str  # user, assistant, system, tool
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    model_used: Optional[str] = None
    cost: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationRequest:
    """Request for text generation."""
    messages: List[Message]
    context: TaskContext
    model_preference: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result from text generation."""
    message: Message
    model_used: str
    cost: float
    response_time: float
    token_usage: Dict[str, int]
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """Current state of an agent."""
    agent_id: str
    conversation_id: str
    memory: List[Message]
    context: TaskContext
    active_tools: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class ModelInterface(ABC):
    """Abstract interface for language models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.metrics = {
            "total_requests": 0,
            "total_cost": 0.0,
            "total_response_time": 0.0,
            "error_count": 0,
            "quality_scores": []
        }
    
    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate a response to the given request."""
        pass
    
    @abstractmethod
    async def stream_generate(self, request: GenerationRequest):
        """Generate a streaming response."""
        pass
    
    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate the cost for a given number of tokens."""
        pass
    
    def update_metrics(self, result: GenerationResult):
        """Update model performance metrics."""
        self.metrics["total_requests"] += 1
        self.metrics["total_cost"] += result.cost
        self.metrics["total_response_time"] += result.response_time
        
        if result.quality_score is not None:
            self.metrics["quality_scores"].append(result.quality_score)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        if self.metrics["total_requests"] == 0:
            return {"avg_response_time": 0.0, "avg_cost": 0.0, "avg_quality": 0.0}
        
        return {
            "avg_response_time": self.metrics["total_response_time"] / self.metrics["total_requests"],
            "avg_cost": self.metrics["total_cost"] / self.metrics["total_requests"],
            "avg_quality": sum(self.metrics["quality_scores"]) / len(self.metrics["quality_scores"]) if self.metrics["quality_scores"] else 0.0,
            "error_rate": self.metrics["error_count"] / self.metrics["total_requests"]
        }


class ModelSelector(ABC):
    """Abstract model selection strategy."""
    
    @abstractmethod
    def select_model(self, 
                    request: GenerationRequest, 
                    available_models: List[ModelInterface]) -> ModelInterface:
        """Select the best model for the given request."""
        pass
    
    @abstractmethod
    def update_performance(self, model: ModelInterface, result: GenerationResult):
        """Update performance data based on generation result."""
        pass


class Tool(ABC):
    """Abstract base class for agent tools."""
    
    def __init__(self, name: str, description: str, config: Dict[str, Any] = None):
        self.name = name
        self.description = description
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        pass
    
    @property
    @abstractmethod
    def schema(self) -> Dict[str, Any]:
        """Return the tool's function schema for LLM function calling."""
        pass


class Agent(ABC):
    """Abstract base class for agents."""
    
    def __init__(self, 
                 agent_id: str,
                 name: str,
                 description: str,
                 capabilities: List[AgentCapability],
                 tools: List[Tool],
                 model_selector: ModelSelector,
                 config: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.tools = {tool.name: tool for tool in tools}
        self.model_selector = model_selector
        self.config = config or {}
        
        # Initialize state
        self.state = AgentState(
            agent_id=agent_id,
            conversation_id=str(uuid.uuid4()),
            memory=[],
            context=TaskContext.CHAT,
            active_tools=list(self.tools.keys())
        )
    
    @abstractmethod
    async def process_message(self, 
                            message: str, 
                            context: TaskContext = TaskContext.CHAT,
                            **kwargs) -> str:
        """Process a user message and return a response."""
        pass
    
    @abstractmethod
    async def execute_task(self, 
                         task: str, 
                         context: TaskContext,
                         **kwargs) -> Dict[str, Any]:
        """Execute a complex task and return structured results."""
        pass
    
    def add_to_memory(self, message: Message):
        """Add a message to the agent's memory."""
        self.state.memory.append(message)
        self.state.updated_at = datetime.now()
        
        # Implement memory management (e.g., sliding window)
        max_memory = self.config.get("max_memory_items", 100)
        if len(self.state.memory) > max_memory:
            self.state.memory = self.state.memory[-max_memory:]
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return [name for name, tool in self.tools.items() if tool.enabled]
    
    async def use_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Use a specific tool."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not available")
        
        if not self.tools[tool_name].enabled:
            raise ValueError(f"Tool '{tool_name}' is disabled")
        
        return await self.tools[tool_name].execute(**kwargs)


class MetricsCollector(ABC):
    """Abstract metrics collection interface."""
    
    @abstractmethod
    def record_generation(self, request: GenerationRequest, result: GenerationResult):
        """Record a generation event."""
        pass
    
    @abstractmethod
    def record_cost(self, model: str, cost: float):
        """Record cost information."""
        pass
    
    @abstractmethod
    def record_performance(self, model: str, response_time: float, success: bool):
        """Record performance metrics."""
        pass
    
    @abstractmethod
    def get_summary(self, timeframe: str = "1h") -> Dict[str, Any]:
        """Get metrics summary for a given timeframe."""
        pass


class QualityEvaluator(ABC):
    """Abstract quality evaluation interface."""
    
    @abstractmethod
    async def evaluate_response(self, 
                              request: GenerationRequest, 
                              response: str) -> float:
        """Evaluate the quality of a response (0.0 to 1.0)."""
        pass
    
    @abstractmethod
    def get_evaluation_criteria(self) -> List[str]:
        """Get list of evaluation criteria used."""
        pass


# Event system for the framework
class Event(BaseModel):
    """Base event class."""
    event_type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = Field(default_factory=dict)


class EventHandler(ABC):
    """Abstract event handler."""
    
    @abstractmethod
    async def handle(self, event: Event):
        """Handle an event."""
        pass


class EventBus:
    """Simple event bus implementation."""
    
    def __init__(self):
        self.handlers: Dict[str, List[EventHandler]] = {}
    
    def subscribe(self, event_type: str, handler: EventHandler):
        """Subscribe a handler to an event type."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
    
    async def publish(self, event: Event):
        """Publish an event to all subscribers."""
        if event.event_type in self.handlers:
            for handler in self.handlers[event.event_type]:
                await handler.handle(event)


# Configuration management
class ConfigManager:
    """Configuration management for the framework."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self._config = {}
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_path:
            import yaml
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set a configuration value."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
