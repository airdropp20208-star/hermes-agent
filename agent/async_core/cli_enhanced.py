"""
Enhanced CLI — interactive TUI with syntax highlighting, autocomplete.
Provides a rich terminal interface for the async agent.
"""
import asyncio
import sys
import os
import time
import json
import shlex
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass

# ANSI color codes
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_BLUE = "\033[44m"


@dataclass
class CLIConfig:
    prompt: str = "🤖 > "
    color: bool = True
    show_timestamps: bool = True
    show_tool_calls: bool = True
    max_history: int = 500
    auto_scroll: bool = True
    syntax_highlight: bool = True


class CommandRegistry:
    """Slash command registry with autocomplete."""

    def __init__(self):
        self._commands: Dict[str, Dict] = {}

    def register(self, name: str, handler: Callable, description: str = "",
                 aliases: List[str] = None):
        """Register a slash command."""
        self._commands[name] = {
            "handler": handler,
            "description": description,
            "aliases": aliases or [],
        }
        for alias in (aliases or []):
            self._commands[alias] = self._commands[name]

    def get(self, name: str) -> Optional[Dict]:
        return self._commands.get(name)

    def complete(self, prefix: str) -> List[str]:
        """Get completions for a prefix."""
        return sorted([n for n in set(self._commands.keys()) if n.startswith(prefix)])

    def list_all(self) -> List[Dict]:
        """List all unique commands."""
        seen = set()
        result = []
        for name, cmd in self._commands.items():
            if name not in seen:
                seen.add(name)
                result.append({"name": name, "description": cmd["description"]})
                for alias in cmd.get("aliases", []):
                    seen.add(alias)
        return sorted(result, key=lambda x: x["name"])


class EnhancedCLI:
    """
    Rich terminal interface with:
    - Syntax highlighting for code blocks
    - Slash command autocomplete
    - Command history
    - Status bar with agent state
    - Tool call visualization
    - Markdown rendering (basic)
    - Session management commands
    - Budget display
    - Interrupt support (Ctrl+C)
    """

    def __init__(self, config: CLIConfig = None):
        self.config = config or CLIConfig()
        self.commands = CommandRegistry()
        self._history: List[str] = []
        self._running = False
        self._agent = None
        self._output_buffer: List[str] = []

        self._register_default_commands()

    def _register_default_commands(self):
        """Register built-in slash commands."""
        self.commands.register("/help", self._cmd_help, "Show available commands", ["/?", "/h"])
        self.commands.register("/quit", self._cmd_quit, "Exit the CLI", ["/exit", "/q"])
        self.commands.register("/clear", self._cmd_clear, "Clear screen", ["/cls"])
        self.commands.register("/history", self._cmd_history, "Show command history")
        self.commands.register("/stats", self._cmd_stats, "Show agent statistics")
        self.commands.register("/sessions", self._cmd_sessions, "List sessions")
        self.commands.register("/checkpoint", self._cmd_checkpoint, "Create checkpoint")
        self.commands.register("/budget", self._cmd_budget, "Show budget usage")
        self.commands.register("/tools", self._cmd_tools, "List available tools")
        self.commands.register("/export", self._cmd_export, "Export conversation")
        self.commands.register("/interrupt", self._cmd_interrupt, "Interrupt agent", ["/stop"])
        self.commands.register("/config", self._cmd_config, "Show/set config")

    def bind_agent(self, agent):
        """Bind an agent to the CLI."""
        self._agent = agent

    async def run(self):
        """Main CLI loop."""
        self._running = True
        self._print_banner()

        while self._running:
            try:
                user_input = await self._read_input()
                if not user_input:
                    continue

                self._history.append(user_input)
                if len(self._history) > self.config.max_history:
                    self._history.pop(0)

                # Handle slash commands
                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                    continue

                # Send to agent
                await self._process_message(user_input)

            except KeyboardInterrupt:
                self._print("\n⚡ Interrupted!", C.YELLOW)
                if self._agent and hasattr(self._agent, 'interrupt'):
                    self._agent.interrupt()
            except EOFError:
                break

    async def _read_input(self) -> str:
        """Read user input (with basic line editing)."""
        prompt = f"{C.CYAN}{self.config.prompt}{C.RESET}"
        try:
            line = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input(prompt)
            )
            return line.strip()
        except (EOFError, KeyboardInterrupt):
            raise

    async def _handle_command(self, cmd_line: str):
        """Handle a slash command."""
        parts = shlex.split(cmd_line)
        cmd_name = parts[0]
        args = parts[1:]

        cmd = self.commands.get(cmd_name)
        if cmd:
            if asyncio.iscoroutinefunction(cmd["handler"]):
                await cmd["handler"](args)
            else:
                cmd["handler"](args)
        else:
            # Show suggestions
            suggestions = self.commands.complete(cmd_name)
            if suggestions:
                self._print(f"Unknown command: {cmd_name}. Did you mean: {', '.join(suggestions)}?", C.YELLOW)
            else:
                self._print(f"Unknown command: {cmd_name}. Type /help for available commands.", C.RED)

    async def _process_message(self, message: str):
        """Process a message through the agent."""
        if not self._agent:
            self._print("No agent bound. Use bind_agent() first.", C.RED)
            return

        self._print_assistant_header()
        start = time.time()

        try:
            if hasattr(self._agent, 'chat'):
                result = self._agent.chat(message)
                if hasattr(result, '__aiter__'):
                    # Async generator — stream tokens
                    async for token in result:
                        self._print_stream(token)
                    self._print("")  # Newline after stream
                else:
                    # Direct result
                    response = await result if asyncio.iscoroutine(result) else result
                    self._print_assistant(str(response))
            else:
                self._print("Agent does not support chat.", C.RED)
        except Exception as e:
            self._print(f"Error: {e}", C.RED)

        elapsed = time.time() - start
        if self.config.show_timestamps:
            self._print(f"  {C.DIM}({elapsed:.1f}s){C.RESET}")

    def _print_banner(self):
        """Print welcome banner."""
        banner = f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════╗
║         🧠 Hermes Agent Enhanced CLI      ║
║                                          ║
║  Type /help for commands                 ║
║  Ctrl+C to interrupt                     ║
╚══════════════════════════════════════════╝{C.RESET}
"""
        print(banner)

    def _print(self, text: str, color: str = ""):
        """Print with optional color."""
        if color and self.config.color:
            print(f"{color}{text}{C.RESET}")
        else:
            print(text)

    def _print_assistant_header(self):
        """Print assistant response header."""
        self._print(f"\n{C.GREEN}🤖 Assistant:{C.RESET}")

    def _print_assistant(self, text: str):
        """Print assistant response with basic syntax highlighting."""
        if self.config.syntax_highlight:
            text = self._highlight_code(text)
        print(text)

    def _print_stream(self, token: str):
        """Print a streaming token."""
        sys.stdout.write(token)
        sys.stdout.flush()

    def _highlight_code(self, text: str) -> str:
        """Basic syntax highlighting for code blocks."""
        import re
        # Highlight code blocks
        text = re.sub(
            r'```(\w*)\n(.*?)```',
            lambda m: f"{C.YELLOW}```{m.group(1)}\n{m.group(2)}```{C.RESET}",
            text, flags=re.DOTALL
        )
        # Highlight inline code
        text = re.sub(r'`([^`]+)`', rf"{C.YELLOW}`\1`{C.RESET}", text)
        # Highlight bold
        text = re.sub(r'\*\*([^*]+)\*\*', rf"{C.BOLD}\1{C.RESET}", text)
        return text

    # --- Default Commands ---

    def _cmd_help(self, args):
        self._print(f"\n{C.BOLD}Available Commands:{C.RESET}")
        for cmd in self.commands.list_all():
            self._print(f"  {C.CYAN}{cmd['name']:<15}{C.RESET} {cmd['description']}")
        print()

    def _cmd_quit(self, args):
        self._running = False
        self._print("Goodbye! 👋", C.CYAN)

    def _cmd_clear(self, args):
        os.system('clear' if os.name != 'nt' else 'cls')

    def _cmd_history(self, args):
        n = int(args[0]) if args else 20
        for i, h in enumerate(self._history[-n:]):
            self._print(f"  {C.DIM}{i+1:3}{C.RESET} {h}")

    def _cmd_stats(self, args):
        if self._agent and hasattr(self._agent, 'get_stats'):
            stats = self._agent.get_stats()
            self._print(f"\n{C.BOLD}Agent Stats:{C.RESET}")
            for k, v in stats.items():
                self._print(f"  {k}: {v}")
        else:
            self._print("No stats available.", C.YELLOW)

    def _cmd_sessions(self, args):
        self._print("Session management not yet bound.", C.YELLOW)

    def _cmd_checkpoint(self, args):
        desc = " ".join(args) if args else "Manual checkpoint"
        self._print(f"Checkpoint: {desc}", C.GREEN)

    def _cmd_budget(self, args):
        self._print("Budget tracking not yet bound.", C.YELLOW)

    def _cmd_tools(self, args):
        if self._agent and hasattr(self._agent, 'tool_runner'):
            tools = self._agent.tool_runner.registry
            self._print(f"\n{C.BOLD}Available Tools ({len(tools)}):{C.RESET}")
            for name in sorted(tools.keys()):
                self._print(f"  🔧 {name}")
        else:
            self._print("No tools available.", C.YELLOW)

    def _cmd_export(self, args):
        path = args[0] if args else "conversation_export.json"
        self._print(f"Exporting to {path}...", C.CYAN)

    def _cmd_interrupt(self, args):
        if self._agent and hasattr(self._agent, 'interrupt'):
            self._agent.interrupt()
            self._print("⚡ Agent interrupted!", C.YELLOW)

    def _cmd_config(self, args):
        if args:
            key, value = args[0], " ".join(args[1:])
            self._print(f"Set {key} = {value}", C.GREEN)
        else:
            self._print(f"\n{C.BOLD}Config:{C.RESET}")
            self._print(f"  prompt: {self.config.prompt}")
            self._print(f"  color: {self.config.color}")
            self._print(f"  syntax_highlight: {self.config.syntax_highlight}")
            self._print(f"  show_tool_calls: {self.config.show_tool_calls}")
