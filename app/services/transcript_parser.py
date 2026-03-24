import re
from dataclasses import dataclass
from typing import List


@dataclass
class ParsedTurn:
    speaker: str
    text: str


@dataclass
class ParsedConversation:
    turns: List[ParsedTurn]
    customer_messages: List[str]
    bot_messages: List[str]
    raw_text: str


class TranscriptParser:
    CUSTOMER_PATTERNS = [r"\*\*P:\*\*\s*(.*?)(?=(\n---|\Z))"]
    BOT_PATTERNS = [
        r"\*\*R:\*\*\s*(.*?)(?=(\n---|\Z))",
        r"R:\*\*\s*(.*?)(?=(\n---|\Z))",
    ]

    def parse(self, text: str) -> ParsedConversation:
        text = (text or "").strip()

        turns: List[ParsedTurn] = []
        customer_messages: List[str] = []
        bot_messages: List[str] = []

        for match in re.finditer(
            r"(\*\*P:\*\*|\*\*R:\*\*|R:\*\*)\s*(.*?)(?=(\n---|\Z))", text, re.DOTALL
        ):
            tag = match.group(1)
            content = match.group(2).strip()

            if tag == "**P:**":
                turns.append(ParsedTurn(speaker="customer", text=content))
                customer_messages.append(content)
            else:
                turns.append(ParsedTurn(speaker="bot", text=content))
                bot_messages.append(content)

        return ParsedConversation(
            turns=turns,
            customer_messages=customer_messages,
            bot_messages=bot_messages,
            raw_text=text,
        )

    def count_exact_repetition(self, customer_messages: List[str]) -> int:
        normalized = [
            self._normalize(m) for m in customer_messages if self._normalize(m)
        ]
        if not normalized:
            return 0

        seen = {}
        repeats = 0
        for msg in normalized:
            seen[msg] = seen.get(msg, 0) + 1

        for _, count in seen.items():
            if count > 1:
                repeats += count - 1

        return repeats

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text
