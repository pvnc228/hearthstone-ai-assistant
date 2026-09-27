"""
Textual-based Terminal User Interface (TUI) for Hearthstone AI Assistant.
Provides a reactive, full-screen tactical HUD inspired by Hearthstone Deck Tracker.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Generator, List, Optional

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, RichLog, Static

from .dispatcher import LiveAdvisorDispatcher, LiveRecommendation
from .event_hub import GameStateUpdate, LiveGameSession

logger = logging.getLogger(__name__)

TUI_CSS = """
Screen {
    background: #090d16;
    color: #e2e8f0;
}

#status_bar {
    height: 3;
    background: #1e293b;
    color: #38bdf8;
    border-bottom: solid #334155;
    padding: 0 1;
    content-align: center middle;
    text-style: bold;
}

#content_grid {
    layout: horizontal;
    height: 1fr;
    padding: 1;
}

#left_column {
    width: 50%;
    height: 100%;
    margin-right: 1;
}

#right_column {
    width: 50%;
    height: 100%;
}

.box_panel {
    background: #0f172a;
    border: round #334155;
    padding: 0 1;
    margin-bottom: 1;
}

#board_panel {
    height: 14;
    border: round #38bdf8;
}

#hand_panel {
    height: 12;
    border: round #34d399;
}

#advisor_panel {
    border: double #f59e0b;
    background: #18182f;
    min-height: 14;
}

#log_panel {
    height: 1fr;
    border: round #475569;
    background: #050811;
}
"""


class BoardWidget(Static):
    """Displays hero stats, mana bar, and minions on both sides of the board."""

    def update_board(self, snapshot: Optional[Any], turn_number: int, mana: int, max_mana: int) -> None:
        if not snapshot:
            self.update(Panel("⏳ Ожидание начала матча...", title="⚔️ Игровой стол", border_style="cyan"))
            return

        f_hero = snapshot.friendly_hero
        o_hero = snapshot.opponent_hero

        f_hp = f_hero.get("health", 30)
        f_armor = f_hero.get("armor", 0)
        o_hp = o_hero.get("health", 30)
        o_armor = o_hero.get("armor", 0)

        f_name = f_hero.get("name", "Игрок")
        o_name = o_hero.get("name", "Оппонент")

        mana_filled = "█" * mana
        mana_empty = "░" * max(0, max_mana - mana)
        mana_bar = f"[{mana_filled}{mana_empty}] {mana}/{max_mana}"

        table = Table(box=None, expand=True, padding=(0, 1))
        table.add_column("Сторона", style="bold", width=12)
        table.add_column("Герой", width=22)
        table.add_column("Существа на столе")

        # Opponent row
        o_board = getattr(snapshot, "opponent_board", [])
        o_minions = ", ".join(f"{m.get('name')}({m.get('attack')}/{m.get('health')})" for m in o_board) or "—"
        o_armor_str = f" [cyan]+{o_armor}[/cyan]" if o_armor else ""
        table.add_row(
            "[red]Оппонент[/red]",
            f"{o_name} ([red]{o_hp} HP[/red]{o_armor_str})",
            f"[dim red]{o_minions}[/dim red]",
        )

        # Friendly row
        f_board = getattr(snapshot, "friendly_board", [])
        f_minions = ", ".join(f"{m.get('name')}({m.get('attack')}/{m.get('health')})" for m in f_board) or "—"
        f_armor_str = f" [cyan]+{f_armor}[/cyan]" if f_armor else ""
        table.add_row(
            "[green]Вы (Игрок)[/green]",
            f"{f_name} ([green]{f_hp} HP[/green]{f_armor_str})",
            f"[dim green]{f_minions}[/dim green]",
        )

        title = f"⚔️ Ход {turn_number}  │  Мана: {mana_bar}"
        self.update(Panel(table, title=title, border_style="cyan"))


class HandWidget(Static):
    """Displays hand cards with 1-indexed zone positions and playability."""

    def update_hand(self, snapshot: Optional[Any], friendly_mana: int) -> None:
        if not snapshot or not getattr(snapshot, "friendly_hand", None):
            self.update(Panel("— Рука пуста —", title="🃏 Карты в руке", border_style="green"))
            return

        table = Table(box=None, expand=True, padding=(0, 1))
        table.add_column("#", width=3, style="bold")
        table.add_column("Карта", width=26)
        table.add_column("Мана", width=6, justify="center")
        table.add_column("Статус", justify="left")

        for idx, card in enumerate(snapshot.friendly_hand, start=1):
            cost = card.get("cost", 0)
            name = card.get("name", "Неизвестная карта")
            pos = card.get("zone_position", idx)

            is_playable = (cost <= friendly_mana)
            status_str = "[green]✓ Доступно[/green]" if is_playable else "[dim]Не хватает маны[/dim]"
            cost_str = f"[cyan]{cost}м[/cyan]" if is_playable else f"[dim]{cost}м[/dim]"
            name_str = f"[bold green]{name}[/bold green]" if is_playable else f"[dim]{name}[/dim]"

            table.add_row(f"{pos}", name_str, cost_str, status_str)

        self.update(Panel(table, title=f"🃏 Карты в руке ({len(snapshot.friendly_hand)} шт.)", border_style="green"))


class AdvisorWidget(Static):
    """Displays real-time tactical advice, lethal warning, and coach notes."""

    def update_recommendation(self, rec: Optional[LiveRecommendation]) -> None:
        if not rec:
            self.update(Panel("⏳ Анализ ситуации на столе...", title="🧠 Тактический тренер", border_style="gold1"))
            return

        lines: List[str] = []

        if rec.is_lethal:
            lines.append(
                f"[bold red on black]🔥 ОБНАРУЖЕН ЛЕТАЛЬНЫЙ УРОН! "
                f"({rec.burst_damage} урона vs {rec.opponent_total_hp} HP оппонента)[/bold red on black]\n"
            )

        lines.append("[bold gold1]🎯 РЕКОМЕНДУЕМЫЙ ПЛАН ДЕЙСТВИЙ:[/bold gold1]")
        icons = ["🥇", "🥈", "🥉"]

        for i, guidance in enumerate(rec.top_guidances[:3]):
            icon = icons[i] if i < len(icons) else "  "
            lines.append(f"  {icon} [bold white]{guidance.formatted_instruction}[/bold white]")

        if rec.coach_note:
            lines.append(f"\n💬 [italic yellow]Совет тренера: {rec.coach_note}[/italic yellow]")

        model_badge = f"[dim]Модель: {rec.model_source} ({rec.latency_ms:.1f} мс)[/dim]"
        lines.append(f"\n{model_badge}")

        border_color = "red" if rec.is_lethal else "gold1"
        self.update(Panel("\n".join(lines), title="🧠 Тактический тренер", border_style=border_color))


class HearthstoneTUIApp(App):
    """Textual TUI Application for Hearthstone AI Assistant."""

    CSS = TUI_CSS
    BINDINGS = [
        ("q", "quit", "Выход"),
        ("c", "clear_log", "Очистить лог"),
    ]

    def __init__(
        self,
        session: LiveGameSession,
        line_generator_factory: Callable[[], Generator[str, None, None]],
        source_label: str = "Live Power.log",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.session = session
        self.line_generator_factory = line_generator_factory
        self.source_label = source_label

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(f"🎮 HEARTHSTONE AI ASSISTANT — LIVE TUI HARNESS │ Источник: {self.source_label}", id="status_bar")

        with Horizontal(id="content_grid"):
            with Vertical(id="left_column"):
                yield BoardWidget(id="board_panel")
                yield HandWidget(id="hand_panel")

            with Vertical(id="right_column"):
                yield AdvisorWidget(id="advisor_panel")
                yield RichLog(id="log_panel", wrap=True, highlight=True, markup=True)

        yield Footer()

    def on_mount(self) -> None:
        """Start the background log consumer worker."""
        board = self.query_one(BoardWidget)
        board.update_board(None, 0, 0, 0)
        advisor = self.query_one(AdvisorWidget)
        advisor.update_recommendation(None)
        self.start_log_worker()

    @work(thread=True)
    def start_log_worker(self) -> None:
        """Background thread streaming log lines into the game session."""
        try:
            line_stream = self.line_generator_factory()
            for update in self.session.ingest_lines(line_stream):
                self.call_from_thread(self.apply_update, update)
        except Exception as e:
            logger.exception("Error in log consumer worker: %s", e)
            self.call_from_thread(self.log_message, f"[red]Ошибка потока логов: {e}[/red]")

    def apply_update(self, update: GameStateUpdate) -> None:
        """Applies a GameStateUpdate to the TUI widgets (main thread)."""
        board = self.query_one(BoardWidget)
        hand = self.query_one(HandWidget)
        advisor = self.query_one(AdvisorWidget)
        log = self.query_one(RichLog)

        if update.snapshot:
            board.update_board(
                update.snapshot,
                update.turn_number,
                update.friendly_mana,
                update.friendly_max_mana,
            )
            hand.update_hand(update.snapshot, update.friendly_mana)

        if update.recommendation:
            advisor.update_recommendation(update.recommendation)

        # Log significant events
        if update.event_type == "GAME_START":
            log.write(f"[bold cyan]⚔️ Начало матча: {update.friendly_hero_name} vs {update.opponent_hero_name}[/bold cyan]")
        elif update.event_type == "TURN_START":
            turn_side = "Ваш ход" if update.is_friendly_turn else "Ход оппонента"
            log.write(f"[bold yellow]👉 Ход {update.turn_number} ({turn_side}, мана: {update.friendly_mana}/{update.friendly_max_mana})[/bold yellow]")
        elif update.event_type == "ACTION" and update.raw_action_desc:
            log.write(f"  [dim]✓ {update.raw_action_desc}[/dim]")
        elif update.event_type == "GAME_OVER":
            icon = "🟢 ПОБЕДА" if update.game_result == "WON" else "🔴 ПОРАЖЕНИЕ"
            log.write(f"[bold white on green]🏁 Матч окончен: {icon} (Ход {update.turn_number})[/bold white on green]")

    def log_message(self, message: str) -> None:
        log = self.query_one(RichLog)
        log.write(message)

    def action_clear_log(self) -> None:
        log = self.query_one(RichLog)
        log.clear()
