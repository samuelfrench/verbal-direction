"""Response classifier — uses Ollama to help route ambiguous voice responses."""

from __future__ import annotations

import logging

import ollama as ollama_client

from verbal_direction.config import OllamaConfig

logger = logging.getLogger(__name__)

ROUTING_PROMPT = """\
You are routing a voice response to the correct Claude Code session.

Active sessions and their pending questions:
{sessions}

User's voice response: "{response}"

Which session is this response most likely intended for?
Respond with ONLY the session name, nothing else.
If you cannot determine the target, respond with "UNKNOWN"."""


class ResponseClassifier:
    """Uses Ollama to route ambiguous voice responses to the correct session."""

    def __init__(self, config: OllamaConfig | None = None) -> None:
        self._config = config or OllamaConfig()
        self._client = ollama_client.Client(host=self._config.host)

    async def route_response(
        self,
        response_text: str,
        sessions: dict[str, str],
    ) -> str | None:
        """Determine which session a voice response is intended for.

        Args:
            response_text: The transcribed voice response.
            sessions: Map of session_name -> pending question text.

        Returns:
            Session name or None if cannot determine.
        """
        if not sessions:
            return None

        if len(sessions) == 1:
            return next(iter(sessions))

        sessions_desc = "\n".join(
            f"- {name}: {question}" for name, question in sessions.items()
        )

        prompt = ROUTING_PROMPT.format(
            sessions=sessions_desc,
            response=response_text,
        )

        try:
            response = self._client.chat(
                model=self._config.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 20},
            )
            result = response["message"]["content"].strip()

            if result == "UNKNOWN":
                return None

            # Validate the result is an actual session name
            for name in sessions:
                if name.lower() == result.lower():
                    return name

            return None

        except Exception as e:
            logger.error("Ollama routing failed: %s", e)
            return None
