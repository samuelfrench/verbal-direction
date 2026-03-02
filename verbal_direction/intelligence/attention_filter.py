"""Attention filter — uses Ollama to classify Claude output."""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Literal

import ollama as ollama_client

from verbal_direction.config import OllamaConfig

logger = logging.getLogger(__name__)

Classification = Literal["question", "permission", "error", "meaningful", "informational"]

CLASSIFICATION_PROMPT = """\
You are classifying output from an AI coding assistant (Claude Code).
Your job is to determine if this output requires the user's attention.

Classify the following text into exactly one category:

- "question" — Claude is asking the user a question or waiting for input/decision
- "permission" — Claude is requesting permission to run a tool or command
- "error" — Claude encountered an error that needs user attention
- "meaningful" — Claude produced a meaningful result worth hearing: a report, summary, analysis, completed feature description, important finding, or final answer. Something the user would want to know about.
- "informational" — Routine status updates, progress logs, "reading file X", "running tests", intermediate steps. Not worth interrupting the user for.

Respond with ONLY the category name, nothing else.

Text to classify:
{text}"""


class AttentionFilter:
    """Classifies Claude output using local Ollama to determine what needs voice attention."""

    def __init__(self, config: OllamaConfig | None = None) -> None:
        self._config = config or OllamaConfig()
        self._client = ollama_client.Client(host=self._config.host)

    def _sync_classify(self, prompt: str) -> str:
        """Synchronous Ollama call — meant to run in a thread executor."""
        response = self._client.chat(
            model=self._config.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 10},
        )
        return response["message"]["content"].strip().lower()

    async def classify(self, text: str) -> Classification:
        """Classify text as question/permission/error/informational."""
        # Truncate very long text to avoid overwhelming the model
        if len(text) > 2000:
            text = text[:2000] + "..."

        prompt = CLASSIFICATION_PROMPT.format(text=text)

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, functools.partial(self._sync_classify, prompt)
            )

            # Normalize the response
            if "question" in result:
                return "question"
            elif "permission" in result:
                return "permission"
            elif "error" in result:
                return "error"
            elif "meaningful" in result:
                return "meaningful"
            else:
                return "informational"

        except Exception as e:
            logger.error("Ollama classification failed: %s", e)
            # Fall back to simple heuristic
            return self._heuristic_classify(text)

    @staticmethod
    def _heuristic_classify(text: str) -> Classification:
        """Simple fallback classifier when Ollama is unavailable."""
        text_lower = text.lower()

        # Question indicators
        if text.rstrip().endswith("?"):
            return "question"
        question_phrases = [
            "which", "should i", "do you want", "would you like",
            "please choose", "select", "what would", "how should",
            "do you prefer", "let me know",
        ]
        if any(phrase in text_lower for phrase in question_phrases):
            return "question"

        # Permission indicators
        permission_phrases = [
            "permission", "allow", "approve", "can i run",
            "execute", "want me to run",
        ]
        if any(phrase in text_lower for phrase in permission_phrases):
            return "permission"

        # Error indicators
        error_phrases = ["error", "failed", "exception", "traceback", "cannot"]
        if any(phrase in text_lower for phrase in error_phrases):
            return "error"

        return "informational"
