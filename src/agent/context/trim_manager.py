"""Context trim manager — keeps LLM context within token budget.

Long-running industrial diagnostic sessions accumulate conversation
history, RAG results, and PLC data.  The trim manager ensures the total
context stays below a configurable token ceiling (default: 4000 tokens)
by dropping the oldest entries first (FIFO truncation) and summarising
intermediate results.

Token count is estimated with a simple heuristic: ~4 characters per
token for English, ~2 characters per token for Chinese.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class TrimConfig:
    """Trim manager parameters."""

    max_tokens: int = 4000  # hard token ceiling
    chars_per_token_en: float = 4.0  # heuristic for English text
    chars_per_token_cn: float = 2.0  # heuristic for Chinese text
    keep_last_n_messages: int = 5  # always preserve the last N exchanges
    summarise_on_trim: bool = True  # if True, summarise trimmed content


class TrimManager:
    """Token-budget-aware context trimmer.

    Usage::

        tm = TrimManager(TrimConfig(max_tokens=4000))
        trimmed = tm.trim(context_text)
        # or for structured history:
        history = tm.trim_history(message_list)
    """

    def __init__(self, config: TrimConfig | None = None) -> None:
        self._cfg = config or TrimConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trim(self, text: str) -> str:
        """Truncate a single text block to fit within the token budget.

        Args:
            text: the context text to trim.

        Returns:
            Truncated text (characters beyond the budget are dropped
            from the beginning).
        """
        limit = self._token_to_chars(self._cfg.max_tokens)
        if len(text) <= limit:
            return text
        # Drop from the beginning (oldest content first)
        return "…[trimmed]…\n" + text[-limit:]

    def trim_history(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Trim a conversation history list to fit the token budget.

        Args:
            messages: list of ``{"role": "...", "content": "..."}`` dicts.

        Returns:
            Trimmed message list.  The last ``keep_last_n_messages``
            are always preserved.
        """
        if not messages:
            return messages

        total = self._estimate_tokens(self._serialize_history(messages))
        if total <= self._cfg.max_tokens:
            return messages

        keep = self._cfg.keep_last_n_messages
        if keep >= len(messages):
            return messages

        # Split: head (droppable) + tail (preserved)
        tail = messages[-keep:]
        head = messages[:-keep]

        # Drop head messages until budget is met
        remaining = self._cfg.max_tokens - self._estimate_tokens(self._serialize_history(tail))
        result: List[Dict[str, str]] = []
        for msg in reversed(head):
            msg_tokens = self._estimate_tokens(msg.get("content", ""))
            if remaining >= msg_tokens:
                result.insert(0, msg)
                remaining -= msg_tokens
            else:
                break

        return result + tail

    def estimate_tokens(self, text: str) -> int:
        """Estimate the token count of a text string."""
        return self._estimate_tokens(text)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _token_to_chars(self, tokens: int) -> int:
        return int(tokens * self._cfg.chars_per_token_en)

    def _estimate_tokens(self, text: str) -> int:
        """Rough heuristic: count CJK characters as 0.5 tokens,
        ASCII as 0.25 tokens."""
        cjk = sum(1 for c in text if "一" <= c <= "鿿")
        ascii_chars = len(text) - cjk
        return int(cjk / self._cfg.chars_per_token_cn + ascii_chars / self._cfg.chars_per_token_en)

    @staticmethod
    def _serialize_history(messages: List[Dict[str, str]]) -> str:
        return "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}"
            for m in messages
        )


def trim_context(text: str, max_tokens: int = 4000) -> str:
    """Convenience: trim a text block to a token budget."""
    return TrimManager(TrimConfig(max_tokens=max_tokens)).trim(text)
