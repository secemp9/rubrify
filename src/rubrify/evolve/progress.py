"""Pretty progress display for rubric evolution.

Implements GEPA's LoggerProtocol with colored output, status symbols,
and a summary table. Uses raw ANSI for colors.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

# ANSI color codes — same pattern as harnify's theme.py
_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"
_DIM = "\x1b[2m"
_GREEN = "\x1b[32m"
_RED = "\x1b[31m"
_YELLOW = "\x1b[33m"
_BLUE = "\x1b[34m"
_CYAN = "\x1b[36m"
_MAGENTA = "\x1b[35m"


def _color(code: str, text: str) -> str:
    """Apply ANSI color if stdout is a TTY."""
    if not sys.stderr.isatty():
        return text
    return f"{code}{text}{_RESET}"


def _bold(text: str) -> str:
    return _color(_BOLD, text)


def _green(text: str) -> str:
    return _color(_GREEN, text)


def _red(text: str) -> str:
    return _color(_RED, text)


def _yellow(text: str) -> str:
    return _color(_YELLOW, text)


def _blue(text: str) -> str:
    return _color(_BLUE, text)


def _cyan(text: str) -> str:
    return _color(_CYAN, text)


def _dim(text: str) -> str:
    return _color(_DIM, text)


@dataclass
class EvolutionProgress:
    """Pretty progress logger for rubric evolution.

    Implements GEPA's LoggerProtocol. Intercepts log messages,
    detects patterns (iteration, acceptance, rejection, scores),
    and formats them with color and structure.
    """
    total_budget: int = 200
    show_proposals: bool = True
    _iteration: int = 0
    _accepted: int = 0
    _rejected: int = 0
    _best_score: float = 0.0
    _start_time: float = field(default_factory=time.time)
    _spinner: object = None

    def log(self, message: str) -> None:
        """GEPA's LoggerProtocol implementation."""
        formatted = self._format(message)
        if formatted is not None:
            sys.stderr.write(formatted + "\n")
            sys.stderr.flush()

    def _format(self, msg: str) -> str | None:
        """Parse and format a GEPA log message."""

        # Iteration start with score
        if "Selected program" in msg and "score:" in msg:
            # "Iteration N: Selected program K score: X.XXX"
            parts = msg.split(":")
            iteration = parts[0].strip()
            self._iteration = int(iteration.split()[1]) if len(iteration.split()) > 1 else self._iteration
            score_part = msg.split("score:")[-1].strip()
            try:
                score = float(score_part)
            except ValueError:
                score = 0.0
            elapsed = time.time() - self._start_time
            return (
                f"\n{_bold(_blue(f'⟳ Iteration {self._iteration}'))} "
                f"{_dim(f'[{elapsed:.0f}s]')} "
                f"parent score: {_bold(f'{score:.3f}')}"
            )

        # Proposed new text
        if "Proposed new text for" in msg:
            component = msg.split("Proposed new text for")[-1].split(":")[0].strip()
            if self.show_proposals:
                # Get the proposed text preview
                text_preview = msg.split(":", 3)[-1].strip()[:80] if ":" in msg else ""
                return f"  {_cyan('◆')} mutating {_bold(component)} {_dim(text_preview + '...')}"
            return f"  {_cyan('◆')} mutating {_bold(component)}"

        # Accepted
        if "Continue to full eval" in msg or "Candidate accepted" in msg:
            self._accepted += 1
            # Extract scores
            if "old_sum=" in msg and "new_sum=" in msg:
                old = msg.split("old_sum=")[1].split(",")[0].split(")")[0]
                new = msg.split("new_sum=")[1].split(",")[0].split(")")[0]
                try:
                    delta = float(new) - float(old)
                    delta_str = f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}"
                    return f"  {_green('✓ accepted')} ({delta_str})"
                except ValueError:
                    pass
            if "is better than" in msg:
                return f"  {_green('✓ accepted')}"
            return f"  {_green('✓ accepted')}"

        # Rejected
        if "skipping" in msg and ("not better" in msg or "rejected" in msg):
            self._rejected += 1
            if "old_sum=" in msg and "new_sum=" in msg:
                old = msg.split("old_sum=")[1].split(",")[0].split(")")[0]
                new = msg.split("new_sum=")[1].split(",")[0].split(")")[0]
                try:
                    delta = float(new) - float(old)
                    delta_str = f"{delta:+.3f}"
                    return f"  {_red('✗ rejected')} ({delta_str})"
                except ValueError:
                    pass
            return f"  {_red('✗ rejected')}"

        # New best found
        if "Found a better program" in msg:
            score_part = msg.split("score")[-1].strip().rstrip(".")
            try:
                score = float(score_part)
                self._best_score = score
                return f"  {_green(_bold(f'★ new best: {score:.3f}'))}"
            except ValueError:
                return f"  {_green(_bold('★ new best'))}"

        # Valset score
        if "Valset score for new program:" in msg:
            score_part = msg.split(":")[-1].strip().split()[0]
            try:
                score = float(score_part)
                return f"  {_dim(f'  valset: {score:.3f}')}"
            except ValueError:
                pass
            return None

        # Base program evaluation
        if "Base program full valset score:" in msg:
            score_part = msg.split("score:")[-1].strip().split()[0]
            try:
                score = float(score_part)
                self._best_score = score
                return (
                    f"{_bold('Seed evaluation')}: {_bold(f'{score:.3f}')}\n"
                    f"{_dim(f'Budget: {self.total_budget} metric calls')}"
                )
            except ValueError:
                pass

        # Skip verbose internal messages
        if any(skip in msg for skip in [
            "Individual valset scores",
            "Objective aggregate",
            "pareto front scores",
            "pareto front aggregate",
            "Updated valset pareto",
            "Updated objective pareto",
            "Best valset aggregate",
            "Best program as per",
            "Best score on valset",
            "Linear pareto front",
            "New program candidate",
            "Objective pareto front",
        ]):
            return None

        # Exception/error
        if "Exception" in msg or "Error" in msg or "Traceback" in msg:
            return f"  {_red('⚠ ' + msg)}"

        # Stop requested
        if "Stop requested" in msg or "shutdown" in msg:
            return f"\n{_yellow('⏹ ' + msg)}"

        # Default: pass through dimmed
        if msg.strip():
            return _dim(f"  {msg}")
        return None

    def summary(self) -> str:
        """Generate a final summary string."""
        elapsed = time.time() - self._start_time
        total = self._accepted + self._rejected
        accept_rate = (self._accepted / total * 100) if total > 0 else 0

        lines = [
            "",
            _bold("=" * 50),
            _bold("  Evolution Summary"),
            _bold("=" * 50),
            f"  Best score:    {_bold(_green(f'{self._best_score:.3f}'))}",
            f"  Iterations:    {total}",
            f"  Accepted:      {_green(str(self._accepted))} ({accept_rate:.0f}%)",
            f"  Rejected:      {_red(str(self._rejected))}",
            f"  Duration:      {elapsed:.1f}s",
            _bold("=" * 50),
        ]
        return "\n".join(lines)
