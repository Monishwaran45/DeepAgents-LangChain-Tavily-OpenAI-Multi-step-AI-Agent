import os
import sys

# Force UTF-8 encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import logging
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.columns import Columns

from src.config import AgentConfig
from src.agent.omni_research_agent import OmniResearchAgent, OmniResearchResponse

# Initialize rich console with safe fallback
console = Console(legacy_windows=False)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


def print_banner(focus_mode: str = "web"):
    banner_text = (
        f"[bold cyan]⚡ DEEP RESEARCH ENGINE[/bold cyan] [dim]• Focus: [bold yellow]{focus_mode.upper()}[/bold yellow][/dim]\n"
        "[dim]Powered by Groq LPU Speed, Tavily Search, Multi-Hop Discovery & Source Cards[/dim]"
    )
    console.print(Panel(banner_text, border_style="cyan", expand=False))


def check_environment():
    """Verify API keys and system configuration."""
    status = AgentConfig.validate()
    table = Table(title="Configuration Diagnostics", border_style="bright_blue")
    table.add_column("Component", style="bold")
    table.add_column("Status")
    table.add_column("Details")

    table.add_row(
        "Groq API Key",
        f"[green]Configured[/green] ({status['groq_key_masked']})" if status["groq_configured"] else "[red]Missing[/red]",
        "Ultra-fast Groq LPU inference"
    )
    table.add_row(
        "Tavily API Key",
        f"[green]Configured[/green] ({status['tavily_key_masked']})" if status["tavily_configured"] else "[red]Missing[/red]",
        "Real-time web & academic retrieval"
    )
    table.add_row(
        "Default Model",
        "[cyan]Active[/cyan]",
        status["model"]
    )
    console.print(table)

    if not status["groq_configured"] or not status["tavily_configured"]:
        console.print(
            "\n[yellow]Note: Please set your valid keys in the .env file before running research:[/yellow]\n"
            "   [bold]GROQ_API_KEY[/bold]=gsk_...\n"
            "   [bold]TAVILY_API_KEY[/bold]=tvly-...\n"
        )
        return False
    return True


def progress_tracker(stage: str, data: dict):
    """Render live search and reasoning steps."""
    if stage == "planning":
        console.print(f"[dim]🧠 {data.get('message')}[/dim]")
    elif stage == "queries_ready":
        queries = data.get("queries", [])
        q_list = ", ".join([f"'{q}'" for q in queries])
        console.print(f"[bold blue]🔍 Searching:[/bold blue] {q_list}")
    elif stage == "deep_dive":
        console.print(f"[dim]⚡ {data.get('message')}[/dim]")
    elif stage == "context_ready":
        console.print(f"[green]✓[/green] Found [bold]{data.get('sources_count')}[/bold] verified sources (~{data.get('estimated_tokens')} tokens)")
    elif stage == "synthesizing":
        console.print(f"[dim]✍️ {data.get('message')}[/dim]")
    elif stage == "verifying":
        console.print(f"[dim]🔬 {data.get('message')}[/dim]")


def render_research_output(response: OmniResearchResponse):
    """Render response with Source Cards, Answer, and Follow-up Questions."""
    console.print("\n" + "─" * 80)
    console.print(f"[bold cyan]🌐 Sources ({len(response.sources)})[/bold cyan]\n")

    # Render Source Cards
    source_cards = []
    for s in response.sources[:8]:
        card = Panel(
            f"[bold blue]{s.title[:45]}[/bold blue]\n[dim]{s.domain}[/dim]",
            title=f"[bold green][{s.index}][/bold green]",
            border_style="dim",
            width=36,
        )
        source_cards.append(card)

    if source_cards:
        console.print(Columns(source_cards))

    console.print("─" * 80 + "\n")

    # Render Answer Markdown with [1], [2] citations
    console.print(Markdown(response.content))
    console.print("\n" + "─" * 80)

    # Render Follow-up / Related Questions
    if response.related_questions:
        console.print("[bold yellow]💡 Related Inquiries:[/bold yellow]")
        for i, q in enumerate(response.related_questions, start=1):
            console.print(f"  [bold cyan][{i}][/bold cyan] {q}")
        console.print("\n[dim]Tip: Type 1, 2, or 3 to investigate a related question, or ask a new inquiry.[/dim]")

    console.print(f"[dim]Grounding: {int(response.grounding_score * 100)}% | Mode: {response.focus_mode.upper()} | Tokens: ~{response.tokens_used_estimate}[/dim]\n")


def run_interactive(session_id: str, focus_mode: str = "web"):
    """Interactive deep research terminal assistant."""
    print_banner(focus_mode=focus_mode)
    if not check_environment():
        return

    agent = OmniResearchAgent(session_id=session_id, focus_mode=focus_mode)
    console.print(f"[bold green]Ready for inquiry![/bold green] Focus: [bold cyan]{focus_mode.upper()}[/bold cyan] | Session: [cyan]{session_id}[/cyan]")
    console.print("[dim]Commands: '/focus <web|academic|finance|coding|deep>' | 'clear' | 'exit'[/dim]\n")

    last_related_questions = []

    while True:
        try:
            user_input = console.input("[bold yellow]Inquire > [/bold yellow]").strip()
            if not user_input:
                continue

            # Command: Exit
            if user_input.lower() in ("exit", "quit"):
                console.print("[cyan]Exiting research session. Session saved.[/cyan]")
                break

            # Command: Clear Memory
            if user_input.lower() == "clear":
                agent.memory_mgr.clear()
                last_related_questions = []
                console.print("[yellow]Session memory cleared.[/yellow]")
                continue

            # Command: Focus mode switch
            if user_input.lower().startswith("/focus"):
                parts = user_input.split()
                if len(parts) > 1 and parts[1].lower() in ("web", "academic", "finance", "coding", "deep", "deep_research"):
                    new_mode = "deep_research" if parts[1].lower() == "deep" else parts[1].lower()
                    agent.focus_mode = new_mode
                    console.print(f"[green]Switched focus mode to:[/green] [bold cyan]{new_mode.upper()}[/bold cyan]")
                else:
                    console.print("[yellow]Usage: /focus <web | academic | finance | coding | deep>[/yellow]")
                continue

            # Quick follow-up selection: typing 1, 2, or 3 selects the related question
            if user_input in ("1", "2", "3") and last_related_questions and len(last_related_questions) >= int(user_input):
                selected_q = last_related_questions[int(user_input) - 1]
                console.print(f"[bold cyan]Exploring Related Question:[/bold cyan] {selected_q}\n")
                user_input = selected_q

            console.print("")
            with console.status("[bold green]Synthesizing deep research report...", spinner="dots"):
                response = agent.research(user_input, progress_callback=progress_tracker)

            last_related_questions = response.related_questions
            render_research_output(response)

        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrupted by user.[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")


def run_single_query(query: str, session_id: str, focus_mode: str = "web"):
    """Single-shot query CLI runner."""
    print_banner(focus_mode=focus_mode)
    if not check_environment():
        return

    agent = OmniResearchAgent(session_id=session_id, focus_mode=focus_mode)
    console.print(f"[bold cyan]Researching:[/bold cyan] {query}\n")

    with console.status("[bold green]Conducting deep research...", spinner="dots"):
        response = agent.research(query, progress_callback=progress_tracker)

    render_research_output(response)


def main():
    parser = argparse.ArgumentParser(description="Deep Research AI Engine powered by Groq and Tavily")
    parser.add_argument("-c", "--chat", action="store_true", help="Launch interactive multi-turn Chat Agent with Rich UI")
    parser.add_argument("-q", "--query", type=str, help="Single research query to execute")
    parser.add_argument("-s", "--session-id", type=str, default="default_session", help="Session ID for persistent memory")
    parser.add_argument("-f", "--focus", type=str, default="web", choices=["web", "academic", "finance", "coding", "deep", "deep_research"], help="Focus mode (web, academic, finance, coding, deep)")
    parser.add_argument("--check", action="store_true", help="Check configuration and API keys")

    args = parser.parse_args()
    focus_mode = "deep_research" if args.focus == "deep" else args.focus

    if args.check:
        print_banner(focus_mode=focus_mode)
        check_environment()
        return

    if args.chat:
        from chat_cli import run_chat_cli
        run_chat_cli(session_id=args.session_id)
        return

    if args.query:
        run_single_query(args.query, args.session_id, focus_mode=focus_mode)
    else:
        run_interactive(args.session_id, focus_mode=focus_mode)


if __name__ == "__main__":
    main()
