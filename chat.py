"""
Interactive CLI for the self-learning agent.

Run:
    python chat.py

Commands inside the chat:
    /memories   -> show everything the agent currently remembers about you
    /forget     -> wipe all stored memories for this user
    /exit       -> quit
"""

from rich.console import Console
from rich.panel import Panel

from memory_agent import SelfLearningAgent

console = Console()


def main():
    console.print(Panel.fit(
        "[bold cyan]Self-Learning AI Agent[/bold cyan] (mem0 + Groq, free tier)\n"
        "Type normally to chat. Commands: /memories  /forget  /exit",
        border_style="cyan",
    ))

    user_id = console.input("[bold]Enter a user id (any name, e.g. 'aditya'): [/bold]").strip() or "default_user"
    agent = SelfLearningAgent(user_id=user_id)

    while True:
        user_input = console.input(f"\n[bold green]{user_id}>[/bold green] ").strip()

        if not user_input:
            continue

        if user_input.lower() == "/exit":
            console.print("[yellow]Goodbye![/yellow]")
            break

        if user_input.lower() == "/memories":
            mems = agent.get_all_memories()
            if not mems:
                console.print("[dim]No memories stored yet.[/dim]")
            else:
                for m in mems:
                    console.print(f"  • {m['memory']}")
            continue

        if user_input.lower() == "/forget":
            agent.forget_everything()
            console.print("[red]All memories for this user have been deleted.[/red]")
            continue

        reply = agent.chat(user_input)
        console.print(f"[bold cyan]agent>[/bold cyan] {reply}")


if __name__ == "__main__":
    main()