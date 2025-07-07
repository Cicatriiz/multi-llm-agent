#!/usr/bin/env python3
"""
Multi-LLM Agent CLI

Command-line interface for the intelligent multi-LLM agentic framework.
Provides easy access to all framework capabilities with intelligent model switching.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
from rich.syntax import Syntax
from rich.markdown import Markdown

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_llm_agent.core.base import ConfigManager, TaskContext
from multi_llm_agent.core.framework import MultiLLMFramework


app = typer.Typer(
    name="mla",
    help="Multi-LLM Agent - Intelligent model switching for any task",
    add_completion=False,
    rich_markup_mode="rich"
)

console = Console()


def load_framework() -> MultiLLMFramework:
    """Load and initialize the framework."""
    # Look for config file
    config_paths = [
        "config/default.yaml",
        "multi_llm_config.yaml",
        os.path.expanduser("~/.multi_llm_agent/config.yaml"),
        "/etc/multi_llm_agent/config.yaml"
    ]
    
    config_path = None
    for path in config_paths:
        if os.path.exists(path):
            config_path = path
            break
    
    if not config_path:
        console.print("[red]No configuration file found. Please create config/default.yaml[/red]")
        raise typer.Exit(1)
    
    config_manager = ConfigManager(config_path)
    config_manager.load_config()
    
    framework = MultiLLMFramework(config_manager)
    return framework


@app.command("chat")
def chat_command(
    message: str = typer.Argument(..., help="Message to send to the agent"),
    context: str = typer.Option("chat", help="Task context (chat, coding, analysis, reasoning, etc.)"),
    model: Optional[str] = typer.Option(None, help="Specific model to use"),
    temperature: Optional[float] = typer.Option(None, help="Temperature for generation"),
    stream: bool = typer.Option(False, help="Stream the response"),
    agent: str = typer.Option("default", help="Agent to use"),
):
    """Chat with the multi-LLM agent."""
    try:
        framework = load_framework()
        
        # Convert context string to TaskContext enum
        try:
            task_context = TaskContext(context.lower())
        except ValueError:
            console.print(f"[red]Invalid context: {context}[/red]")
            console.print("Valid contexts: " + ", ".join([c.value for c in TaskContext]))
            raise typer.Exit(1)
        
        async def run_chat():
            agent_instance = await framework.get_agent(agent)
            
            if stream:
                console.print(f"[blue]Agent ({agent}):[/blue]", end=" ")
                async for chunk in agent_instance.stream_response(message, task_context, model_preference=model, temperature=temperature):
                    console.print(chunk, end="")
                console.print()  # New line at the end
            else:
                with console.status("[bold green]Thinking..."):
                    response = await agent_instance.process_message(
                        message, 
                        task_context, 
                        model_preference=model, 
                        temperature=temperature
                    )
                
                # Display response with syntax highlighting if it contains code
                if "```" in response:
                    console.print(Markdown(response))
                else:
                    console.print(f"[blue]Agent ({agent}):[/blue] {response}")
        
        asyncio.run(run_chat())
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("code")
def code_command(
    task: str = typer.Argument(..., help="Coding task description"),
    language: str = typer.Option("python", help="Programming language"),
    model: Optional[str] = typer.Option(None, help="Specific model to use"),
    save_to: Optional[str] = typer.Option(None, help="File to save the generated code"),
    execute: bool = typer.Option(False, help="Execute the generated code"),
):
    """Generate code with intelligent model selection."""
    try:
        framework = load_framework()
        
        async def run_code_generation():
            agent = await framework.get_agent("coder")
            
            with console.status("[bold green]Generating code..."):
                result = await agent.execute_task(
                    task, 
                    TaskContext.CODING,
                    language=language,
                    model_preference=model
                )
            
            # Extract code from result
            code = result.get("code", "")
            explanation = result.get("explanation", "")
            
            # Display results
            console.print(Panel(explanation, title="Explanation"))
            
            if code:
                syntax = Syntax(code, language, theme="monokai", line_numbers=True)
                console.print(Panel(syntax, title=f"{language.title()} Code"))
                
                # Save to file if requested
                if save_to:
                    with open(save_to, 'w') as f:
                        f.write(code)
                    console.print(f"[green]Code saved to {save_to}[/green]")
                
                # Execute if requested
                if execute and language == "python":
                    console.print("\n[yellow]Executing code...[/yellow]")
                    try:
                        exec(code)
                    except Exception as e:
                        console.print(f"[red]Execution error: {e}[/red]")
        
        asyncio.run(run_code_generation())
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("analyze")
def analyze_command(
    content: str = typer.Argument(..., help="Content to analyze"),
    analysis_type: str = typer.Option("general", help="Type of analysis"),
    model: Optional[str] = typer.Option(None, help="Specific model to use"),
    format: str = typer.Option("text", help="Output format (text, json, markdown)"),
):
    """Analyze content with intelligent model selection."""
    try:
        framework = load_framework()
        
        async def run_analysis():
            agent = await framework.get_agent("analyst")
            
            with console.status("[bold green]Analyzing..."):
                result = await agent.execute_task(
                    content,
                    TaskContext.ANALYSIS,
                    analysis_type=analysis_type,
                    model_preference=model
                )
            
            # Format and display results
            if format == "json":
                console.print_json(json.dumps(result, indent=2))
            elif format == "markdown":
                console.print(Markdown(result.get("analysis", str(result))))
            else:
                console.print(Panel(result.get("analysis", str(result)), title="Analysis Results"))
        
        asyncio.run(run_analysis())
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("solve")
def solve_command(
    problem: str = typer.Argument(..., help="Problem to solve"),
    approach: str = typer.Option("adaptive", help="Solving approach (adaptive, collaborative, step_by_step)"),
    max_iterations: Optional[int] = typer.Option(None, help="Maximum iterations for complex solving"),
    models: Optional[str] = typer.Option(None, help="Comma-separated list of models to use"),
):
    """Solve complex problems using multi-LLM approaches."""
    try:
        framework = load_framework()
        
        async def run_solver():
            agent = await framework.get_agent("reasoner")
            
            model_list = models.split(",") if models else None
            
            with console.status("[bold green]Solving problem..."):
                if approach == "collaborative":
                    result = await agent.collaborative_solve(problem, model_list)
                elif approach == "step_by_step":
                    # For step-by-step, we need to break down the problem
                    steps = [
                        "Understand the problem",
                        "Identify key components",
                        "Develop solution strategy",
                        "Implement solution",
                        "Verify and refine"
                    ]
                    result = await agent.solve_step_by_step(problem, steps)
                else:  # adaptive
                    result = await agent.solve_complex_problem(
                        problem, 
                        max_iterations=max_iterations
                    )
            
            # Display results
            console.print(Panel(problem, title="Problem"))
            
            if "final_solution" in result:
                console.print(Panel(result["final_solution"], title="Solution"))
            elif "consensus_solution" in result:
                console.print(Panel(result["consensus_solution"], title="Consensus Solution"))
            
            # Show performance metrics
            if "cost_summary" in result and result["cost_summary"]:
                cost_table = Table(title="Cost Summary")
                cost_table.add_column("Model")
                cost_table.add_column("Cost", justify="right")
                
                for model, cost in result["cost_summary"]["costs_by_model"].items():
                    cost_table.add_row(model, f"${cost:.4f}")
                
                console.print(cost_table)
        
        asyncio.run(run_solver())
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("benchmark")
def benchmark_command(
    task: str = typer.Argument(..., help="Task to benchmark"),
    models: Optional[str] = typer.Option(None, help="Comma-separated list of models to benchmark"),
    iterations: int = typer.Option(3, help="Number of iterations per model"),
    context: str = typer.Option("general", help="Task context"),
):
    """Benchmark models on a specific task."""
    try:
        framework = load_framework()
        
        async def run_benchmark():
            model_list = models.split(",") if models else None
            
            with console.status("[bold green]Running benchmark..."):
                results = await framework.benchmark_models(
                    task,
                    TaskContext(context),
                    model_list,
                    iterations
                )
            
            # Display benchmark results
            table = Table(title="Benchmark Results")
            table.add_column("Model")
            table.add_column("Avg Time (s)", justify="right")
            table.add_column("Avg Cost ($)", justify="right")
            table.add_column("Success Rate", justify="right")
            table.add_column("Errors", justify="right")
            
            for model_name, model_results in results.items():
                avg_time = model_results.get("avg_time", 0)
                avg_cost = model_results.get("avg_cost", 0)
                success_rate = 1 - (model_results.get("errors", 0) / iterations)
                errors = model_results.get("errors", 0)
                
                table.add_row(
                    model_name,
                    f"{avg_time:.2f}",
                    f"{avg_cost:.4f}",
                    f"{success_rate:.2%}",
                    str(errors)
                )
            
            console.print(table)
        
        asyncio.run(run_benchmark())
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("status")
def status_command():
    """Show framework status and model performance."""
    try:
        framework = load_framework()
        
        async def show_status():
            status = await framework.get_status()
            
            # Show available models
            models_table = Table(title="Available Models")
            models_table.add_column("Model")
            models_table.add_column("Provider")
            models_table.add_column("Type")
            models_table.add_column("Status")
            models_table.add_column("Requests", justify="right")
            models_table.add_column("Avg Cost", justify="right")
            
            for model_name, model_info in status.get("models", {}).items():
                models_table.add_row(
                    model_name,
                    model_info.get("provider", "Unknown"),
                    model_info.get("type", "Unknown"),
                    "[green]Available[/green]" if model_info.get("available", False) else "[red]Unavailable[/red]",
                    str(model_info.get("total_requests", 0)),
                    f"${model_info.get('avg_cost', 0):.4f}"
                )
            
            console.print(models_table)
            
            # Show performance summary
            if "performance_summary" in status:
                perf_summary = status["performance_summary"]
                console.print(f"\n[bold]Total Requests:[/bold] {perf_summary.get('total_requests', 0)}")
                console.print(f"[bold]Total Cost:[/bold] ${perf_summary.get('total_cost', 0):.4f}")
                console.print(f"[bold]Average Response Time:[/bold] {perf_summary.get('avg_response_time', 0):.2f}s")
        
        asyncio.run(show_status())
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("config")
def config_command(
    action: str = typer.Argument(..., help="Action to perform (show, set, validate)"),
    key: Optional[str] = typer.Option(None, help="Configuration key"),
    value: Optional[str] = typer.Option(None, help="Configuration value"),
):
    """Manage framework configuration."""
    try:
        framework = load_framework()
        config = framework.config
        
        if action == "show":
            if key:
                val = config.get(key)
                console.print(f"{key}: {val}")
            else:
                # Show full config (redacted)
                full_config = config._config
                # Redact sensitive information
                redacted_config = _redact_sensitive_info(full_config)
                console.print_json(json.dumps(redacted_config, indent=2))
        
        elif action == "set":
            if not key or not value:
                console.print("[red]Both key and value are required for set action[/red]")
                raise typer.Exit(1)
            
            # Try to parse value as JSON, fallback to string
            try:
                parsed_value = json.loads(value)
            except json.JSONDecodeError:
                parsed_value = value
            
            config.set(key, parsed_value)
            console.print(f"[green]Set {key} = {parsed_value}[/green]")
        
        elif action == "validate":
            # Validate configuration
            is_valid, errors = _validate_config(config._config)
            if is_valid:
                console.print("[green]Configuration is valid[/green]")
            else:
                console.print("[red]Configuration has errors:[/red]")
                for error in errors:
                    console.print(f"  - {error}")
        
        else:
            console.print(f"[red]Unknown action: {action}[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("interactive")
def interactive_command():
    """Start interactive chat session."""
    try:
        framework = load_framework()
        
        async def interactive_session():
            agent = await framework.get_agent("default")
            
            console.print("[bold blue]Multi-LLM Agent Interactive Session[/bold blue]")
            console.print("Type 'quit' or 'exit' to end the session")
            console.print("Type '/help' for available commands\n")
            
            while True:
                try:
                    user_input = console.input("[bold green]You:[/bold green] ")
                    
                    if user_input.lower() in ['quit', 'exit']:
                        break
                    
                    if user_input.startswith('/'):
                        await _handle_interactive_command(user_input, framework)
                        continue
                    
                    with console.status("[bold blue]Thinking..."):
                        response = await agent.process_message(user_input, TaskContext.CHAT)
                    
                    console.print(f"[bold blue]Agent:[/bold blue] {response}\n")
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
            
            console.print("[yellow]Session ended.[/yellow]")
        
        asyncio.run(interactive_session())
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


async def _handle_interactive_command(command: str, framework):
    """Handle interactive session commands."""
    if command == "/help":
        console.print("""
[bold]Available Commands:[/bold]
/help - Show this help message
/status - Show framework status
/models - List available models
/context <context> - Change task context
/agent <agent> - Switch to different agent
/clear - Clear conversation history
""")
    elif command == "/status":
        status = await framework.get_status()
        console.print(f"Active models: {len(status.get('models', {}))}")
        console.print(f"Total requests: {status.get('total_requests', 0)}")
    
    elif command == "/models":
        status = await framework.get_status()
        for model_name in status.get("models", {}):
            console.print(f"  - {model_name}")
    
    elif command.startswith("/context"):
        parts = command.split()
        if len(parts) > 1:
            console.print(f"Context switched to: {parts[1]}")
        else:
            console.print("Usage: /context <context_name>")
    
    elif command.startswith("/agent"):
        parts = command.split()
        if len(parts) > 1:
            console.print(f"Agent switched to: {parts[1]}")
        else:
            console.print("Usage: /agent <agent_name>")
    
    elif command == "/clear":
        console.print("Conversation history cleared.")
    
    else:
        console.print(f"Unknown command: {command}")


def _redact_sensitive_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """Redact sensitive information from configuration."""
    sensitive_keys = ["api_key", "secret", "password", "token"]
    
    def redact_dict(d):
        if isinstance(d, dict):
            return {
                k: "***REDACTED***" if any(sensitive in k.lower() for sensitive in sensitive_keys)
                else redact_dict(v) if isinstance(v, (dict, list)) else v
                for k, v in d.items()
            }
        elif isinstance(d, list):
            return [redact_dict(item) for item in d]
        return d
    
    return redact_dict(config)


def _validate_config(config: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate configuration."""
    errors = []
    
    # Check required sections
    required_sections = ["models", "selection", "api"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Validate models section
    if "models" in config:
        for provider, models in config["models"].items():
            if not isinstance(models, dict):
                errors.append(f"Models for provider {provider} should be a dictionary")
                continue
            
            for model_name, model_config in models.items():
                if "api_name" not in model_config:
                    errors.append(f"Model {model_name} missing api_name")
    
    return len(errors) == 0, errors


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
