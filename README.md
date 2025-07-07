# Multi-LLM Agent Framework

🤖 **Intelligent Multi-LLM Agentic Framework with Adaptive Model Switching**

An advanced framework inspired by AB-MCTS research that intelligently switches between multiple language models to optimize performance, cost, and quality for any task.

## ✨ Features

- **🧠 Intelligent Model Selection**: AB-MCTS inspired adaptive model switching based on context, performance, and cost
- **🔄 Multi-LLM Support**: Latest models from OpenAI (o4-mini, o3, gpt-4.1), Google (Gemini 2.5), DeepSeek (R1, Chat)
- **🎯 Context-Aware**: Automatically selects optimal models for coding, analysis, reasoning, chat, and creative tasks
- **📊 Performance Tracking**: Real-time metrics collection and analysis for continuous optimization
- **🛠️ Agentic Tools**: Built-in tools for web search, code execution, file operations, and data analysis
- **⚡ Async & Streaming**: Full async support with streaming responses for real-time interaction
- **🎨 Rich CLI**: Beautiful command-line interface with syntax highlighting and interactive mode
- **💰 Cost Optimization**: Intelligent cost management with spending limits and budget tracking
- **🔌 Extensible**: Plugin architecture for custom models, tools, and evaluation metrics

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/multi-llm-agent.git
cd multi-llm-agent

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Configuration

Create your configuration file:

```bash
cp config/default.yaml config/local.yaml
```

Set your API keys:

```bash
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
```

### Basic Usage

```bash
# Simple chat
mla chat "What's the capital of France?"

# Code generation
mla code "Create a Python function to calculate fibonacci numbers"

# Complex problem solving
mla solve "How can I optimize my database queries for better performance?" --approach collaborative

# Analysis
mla analyze "This is some text to analyze" --analysis-type technical

# Interactive mode
mla interactive
```

## 🧠 Intelligent Model Selection

The framework uses an AB-MCTS inspired approach to intelligently select the best model for each task:

### Context-Aware Selection

- **Coding Tasks**: Prefers `gpt-4.1`, `deepseek-chat` for their coding optimization
- **Reasoning Tasks**: Uses `o3`, `o4-mini`, `deepseek-reasoner` for complex logic
- **Analysis**: Leverages `o3`, `gemini-2.5-pro` for comprehensive analysis
- **Chat**: Optimizes for speed with `gemini-2.5-flash`, `gpt-4.1-nano`
- **Creative**: Utilizes `gemini-2.5-pro`, `o3` for creative tasks

### Performance Learning

The framework continuously learns from:
- Response quality scores
- Response times
- Cost efficiency
- Success rates
- User feedback

### Adaptive Strategies

1. **Exploration vs Exploitation**: Balances trying new models vs using proven performers
2. **Context Memory**: Remembers which models work best for specific contexts
3. **Cost Awareness**: Considers budget constraints in model selection
4. **Performance Trends**: Adapts to changing model performance over time

## 🛠️ Advanced Usage

### Multi-Model Collaboration

```bash
# Collaborative problem solving with multiple models
mla solve "Design a scalable microservices architecture" \
  --approach collaborative \
  --models "o3,gemini-2.5-pro,deepseek-reasoner"
```

### Step-by-Step Problem Solving

```bash
# Break down complex problems into steps
mla solve "Build a machine learning pipeline for fraud detection" \
  --approach step_by_step
```

### Benchmarking

```bash
# Compare model performance on specific tasks
mla benchmark "Write a sorting algorithm in Python" \
  --models "gpt-4.1,deepseek-chat,o4-mini" \
  --iterations 5
```

### Custom Agents

```python
from multi_llm_agent.core.base import Agent, TaskContext
from multi_llm_agent.core.framework import MultiLLMFramework

# Create custom specialized agent
class SecurityAgent(Agent):
    async def process_message(self, message: str, context: TaskContext, **kwargs):
        # Add security-specific prompting and tools
        security_context = f"As a cybersecurity expert: {message}"
        return await super().process_message(security_context, context, **kwargs)

# Use in framework
framework = MultiLLMFramework(config)
framework.register_agent("security", SecurityAgent)
```

## 📊 Performance Monitoring

### Real-time Metrics

```bash
# View current performance status
mla status

# Get detailed performance breakdown
mla config show metrics
```

### Cost Tracking

The framework automatically tracks:
- Per-model costs
- Daily/monthly spending
- Cost per task type
- Budget alerts and limits

### Quality Metrics

Continuous evaluation of:
- Response coherence
- Task completion success
- User satisfaction scores
- Context appropriateness

## 🔧 Configuration

### Model Configuration

```yaml
models:
  openai:
    o4_mini:
      name: "o4-mini"
      api_name: "o4-mini"
      contexts: ["reasoning", "general"]
      cost_per_1k_input: 0.0011
      cost_per_1k_output: 0.0044
      model_type: "reasoning"
```

### Selection Strategy

```yaml
selection:
  strategy: "adaptive_ab_mcts"
  context_preferences:
    coding:
      primary: ["gpt_4_1", "deepseek_chat"]
      fallback: ["o4_mini", "gemini_2_5_pro"]
      temperature_override: 0.3
```

### AB-MCTS Parameters

```yaml
ab_mcts:
  enabled: true
  max_iterations: 25
  exploration_constant: 1.4
  selection_temperature: 0.1
  model_combinations:
    high_performance:
      models: ["o4_mini", "gemini_2_5_pro", "deepseek_reasoner"]
```

## 🔌 Extensions & Tools

### Built-in Tools

- **Web Search**: Real-time information retrieval
- **Code Execution**: Safe sandboxed code running
- **File Operations**: Read/write/manipulate files
- **Data Analysis**: pandas, numpy, visualization
- **Math Solver**: Symbolic and numeric computation

### Custom Tools

```python
from multi_llm_agent.core.base import Tool

class DatabaseTool(Tool):
    async def execute(self, query: str, **kwargs):
        # Execute database query
        return {"results": [...]}
    
    @property
    def schema(self):
        return {
            "name": "database_query",
            "description": "Execute SQL queries",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query"}
                }
            }
        }
```

## 🌐 API & Integration

### REST API

```bash
# Start the API server
mla-server

# Make requests
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "context": "chat"}'
```

### Python API

```python
import asyncio
from multi_llm_agent import MultiLLMFramework, TaskContext

async def main():
    framework = MultiLLMFramework.from_config("config/default.yaml")
    agent = await framework.get_agent("default")
    
    response = await agent.process_message(
        "Explain quantum computing", 
        TaskContext.ANALYSIS
    )
    print(response)

asyncio.run(main())
```

### Web Interface

```bash
# Launch web UI
mla-web
```

## 🧪 Research & Benchmarks

### AB-MCTS Integration

This framework implements concepts from the AB-MCTS research paper:

- **Adaptive Branching**: Dynamic model selection trees
- **Multi-Model Consensus**: Collaborative problem solving
- **Performance-Based Selection**: Learning from outcomes
- **Exploration Strategies**: Balancing known vs unknown models

### Benchmark Results

Performance benchmarks can be run using the built-in benchmarking tools:

```bash
# Run benchmarks on your tasks
mla benchmark "Your task here" --models "o4-mini,gpt-4.1,deepseek-chat" --iterations 5
```

| Task Type | Single Best Model | Multi-LLM Framework | Status |
|-----------|------------------|-------------------|--------|
| Coding    | Run benchmark    | Run benchmark     | 📊 [Benchmark yourself](docs/benchmarking.md) |
| Analysis  | Run benchmark    | Run benchmark     | 📊 [Benchmark yourself](docs/benchmarking.md) |
| Reasoning | Run benchmark    | Run benchmark     | 📊 [Benchmark yourself](docs/benchmarking.md) |
| Cost Efficiency | Run benchmark | Run benchmark  | 📊 [Benchmark yourself](docs/benchmarking.md) |

> **Note**: Actual performance will vary based on your specific tasks, model availability, and configuration. We encourage you to run your own benchmarks with your real-world use cases.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
uv sync --group dev

# Run tests
pytest

# Code formatting
black src/
isort src/

# Type checking
mypy src/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the AB-MCTS research from SakanaAI
- Built on the excellent work of the open-source AI community
- Special thanks to the teams at OpenAI, Google, and DeepSeek for their amazing models

## 📚 Documentation

For more detailed documentation, visit our [docs](docs/) directory:

- [Architecture Overview](docs/architecture.md)
- [Model Selection Guide](docs/model-selection.md)
- [Tool Development](docs/tools.md)
- [Performance Tuning](docs/performance.md)
- [API Reference](docs/api.md)

## 🔮 Roadmap

- [ ] Additional model providers (Claude, Cohere, etc.)
- [ ] Advanced reasoning chains (CoT, ToT)
- [ ] Multi-modal support (vision, audio)
- [ ] Distributed agent networks
- [ ] Fine-tuning integration
- [ ] Enterprise features (SSO, audit logs)
- [ ] GUI application
- [ ] Model marketplace integration

---

**Built with ❤️ for the AI community**
