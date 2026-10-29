import os
import sys

# Force UTF-8 encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
import argparse
from typing import Optional
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt

from src.config import AgentConfig
from src.agent.chat_agent import ChatAgent

console = Console(legacy_windows=False)


def render_header(model: str, session_id: str, auto_search: bool):
    """Render terminal chat header banner."""
    search_badge = "[bold green]ON[/bold green]" if auto_search else "[dim]OFF[/dim]"
    banner_content = (
        f"[bold cyan]🤖 AI CHAT ASSISTANT[/bold cyan] [dim]• Powered by Groq LPU & Tavily Web Grounding[/dim]\n"
        f"[dim]Model:[/dim] [bold yellow]{model}[/bold yellow] | "
        f"[dim]Session:[/dim] [bold magenta]{session_id}[/bold magenta] | "
        f"[dim]Live Search:[/dim] {search_badge}\n"
        "[dim]Type your message or [/dim][bold cyan]/help[/bold cyan][dim] for commands, [/dim][bold cyan]/exit[/bold cyan][dim] to quit.[/dim]"
    )
    console.print(Panel(banner_content, border_style="cyan", padding=(0, 2)))


def show_help():
    """Display interactive commands table."""
    table = Table(title="Interactive Chat Commands", border_style="bright_blue")
    table.add_column("Command", style="bold cyan")
    table.add_column("Description")

    table.add_row("/search", "Toggle automatic live web search ON/OFF")
    table.add_row("/force <query>", "Force Tavily web search for this specific question")
    table.add_row("/model <name>", "Switch model (e.g. openai/gpt-oss-120b, qwen/qwen3.8-27b)")
    table.add_row("/history", "View past conversation turns in this session")
    table.add_row("/clear", "Reset conversation memory for this session")
    table.add_row("/session <name>", "Switch to or load a different session ID")
    table.add_row("/export <file.md>", "Export conversation transcript to a markdown file")
    table.add_row("/help", "Show this commands reference")
    table.add_row("/exit, /quit", "Save and exit chat")
    console.print(table)


def view_history(agent: ChatAgent):
    """Render chat history in a formatted table."""
    history = agent.memory_mgr.history
    if not history:
        console.print("[dim]No messages in current session history.[/dim]\n")
        return

    table = Table(title=f"Session History: {agent.session_id}", border_style="dim")
    table.add_column("#", style="bold", width=4)
    table.add_column("User Query", style="yellow")
    table.add_column("Agent Response Summary", style="dim")
    table.add_column("Sources", style="cyan", width=8)

    for idx, turn in enumerate(history, start=1):
        table.add_row(
            str(idx),
            turn.user_query[:50] + ("..." if len(turn.user_query) > 50 else ""),
            turn.agent_response[:70] + ("..." if len(turn.agent_response) > 70 else ""),
            str(turn.citations_count),
        )
    console.print(table)


def export_chat(agent: ChatAgent, filepath: str):
    """Export current chat transcript to Markdown."""
    history = agent.memory_mgr.history
    if not history:
        console.print("[yellow]No conversation history to export.[/yellow]\n")
        return

    if not filepath.endswith(".md"):
        filepath += ".md"

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Chat Session Transcript: {agent.session_id}\n")
            f.write(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n")
            for idx, turn in enumerate(history, start=1):
                f.write(f"### User [{idx}]:\n{turn.user_query}\n\n")
                f.write(f"### Assistant [{idx}]:\n{turn.agent_response}\n\n---\n\n")
        console.print(f"[green]✓ Exported conversation transcript to:[/green] [bold cyan]{filepath}[/bold cyan]\n")
    except Exception as e:
        console.print(f"[red]Failed to export: {e}[/red]\n")


def run_chat_cli(session_id: str = "cli_chat", model_name: Optional[str] = None):
    """Main Rich UI interactive chat loop."""
    model = model_name or AgentConfig.DEFAULT_GROQ_MODEL
    agent = ChatAgent(model_name=model, session_id=session_id)

    render_header(agent.model_name, agent.session_id, agent.auto_search)

    # If previous session had messages, notify user
    if agent.memory_mgr.history:
        console.print(f"[dim]Loaded {len(agent.memory_mgr.history)} previous messages from session '{session_id}'.[/dim]\n")

    while True:
        try:
            # User Prompt input
            user_input = Prompt.ask("\n[bold yellow]You[/bold yellow]").strip()
            if not user_input:
                continue

            # Command: Exit
            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                console.print("[cyan]Exiting chat session. Conversation saved automatically.[/cyan]")
                break

            # Command: Help
            if user_input.lower() in ("/help", "help"):
                show_help()
                continue

            # Command: Toggle Search
            if user_input.lower() == "/search":
                agent.auto_search = not agent.auto_search
                status_txt = "[bold green]ON[/bold green]" if agent.auto_search else "[bold red]OFF[/bold red]"
                console.print(f"[dim]Live search auto-routing is now[/dim] {status_txt}")
                continue

            # Command: Clear Memory
            if user_input.lower() == "/clear":
                agent.memory_mgr.clear()
                console.print("[yellow]Session history cleared.[/yellow]")
                continue

            # Command: View History
            if user_input.lower() == "/history":
                view_history(agent)
                continue

            # Command: Switch Model
            if user_input.lower().startswith("/model"):
                parts = user_input.split()
                if len(parts) > 1:
                    new_model = parts[1].strip()
                    agent.model_name = new_model
                    agent.llm = agent.llm.with_config({"model": new_model})
                    console.print(f"[green]Switched model to:[/green] [bold cyan]{new_model}[/bold cyan]")
                else:
                    console.print("[yellow]Usage: /model <model_name> (e.g. openai/gpt-oss-120b, qwen/qwen3.8-27b)[/yellow]")
                continue

            # Command: Switch Session
            if user_input.lower().startswith("/session"):
                parts = user_input.split()
                if len(parts) > 1:
                    new_session = parts[1].strip()
                    agent = ChatAgent(model_name=agent.model_name, session_id=new_session, auto_search=agent.auto_search)
                    console.print(f"[green]Switched to session:[/green] [bold cyan]{new_session}[/bold cyan]")
                    render_header(agent.model_name, agent.session_id, agent.auto_search)
                else:
                    console.print("[yellow]Usage: /session <session_id>[/yellow]")
                continue

            # Command: Export
            if user_input.lower().startswith("/export"):
                parts = user_input.split(maxsplit=1)
                filepath = parts[1].strip() if len(parts) > 1 else f"chat_export_{agent.session_id}.md"
                export_chat(agent, filepath)
                continue

            # Force search mode: /force <query>
            force_search = False
            if user_input.lower().startswith("/force"):
                force_search = True
                user_input = user_input[6:].strip()
                if not user_input:
                    console.print("[yellow]Usage: /force <query to search>[/yellow]")
                    continue

            # Display User Message Panel
            console.print(Panel(user_input, title="[bold blue]You[/bold blue]", border_style="blue", padding=(0, 1)))

            # Spinner with progress reporting
            def progress(stage: str, data: dict):
                if stage == "searching":
                    console.print(f"[dim]🌐 Searching Tavily for: [italic]{data.get('query')}[/italic]...[/dim]")
                elif stage == "context_ready":
                    console.print(f"[dim]✓ Retained {data.get('count')} sources for grounding[/dim]")

            start_time = time.time()
            with console.status(f"[bold cyan]Thinking with {agent.model_name}...", spinner="dots"):
                response = agent.chat(user_input, force_search=force_search, progress_callback=progress)
            elapsed = round(time.time() - start_time, 2)

            # Display Assistant Response Panel
            sources_footer = ""
            if response.sources:
                sources_links = " | ".join([f"[cyan][{s['index']}][/cyan] [underline]{s['title'][:35]}[/underline]" for s in response.sources[:4]])
                sources_footer = f"\n\n[dim]Sources:[/dim] {sources_links}"

            panel_text = Markdown(response.content)
            console.print(
                Panel(
                    panel_text,
                    title=f"[bold cyan]Assistant ({agent.model_name})[/bold cyan] [dim]({elapsed}s)[/dim]",
                    subtitle=sources_footer if sources_footer else None,
                    border_style="cyan",
                    padding=(1, 2),
                )
            )

        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrupted.[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]\n")


def main():
    parser = argparse.ArgumentParser(description="Conversational AI Chat Agent in Terminal with Rich UI")
    parser.add_argument("-s", "--session-id", type=str, default="cli_chat", help="Session ID for conversation persistence")
    parser.add_argument("-m", "--model", type=str, default=None, help="Groq model name")
    args = parser.parse_args()

    run_chat_cli(session_id=args.session_id, model_name=args.model)


if __name__ == "__main__":
    main()
