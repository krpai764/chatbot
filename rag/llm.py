"""
LLM module for the DataIntern RAG Engine.

Provides a GeminiLLM wrapper around the Google GenAI SDK for generating
text and structured JSON responses using Gemini models.
"""

import json
import logging
import re

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# System instruction used for RAG-grounded answers.
RAG_SYSTEM_INSTRUCTION = """You are DataIntern, an AI assistant specializing in CRM data analysis.

Rules:
1. For CRM data questions, answer ONLY from the provided context. Never hallucinate numbers or facts.
2. For conversational or meta questions (e.g. 'what did I ask before?', 'summarize our conversation', 'what is your name?'), answer naturally using the conversation history provided.
3. If CRM information cannot be found in the context, respond exactly: "I couldn't find this information in the provided documents."
4. Always cite your sources with specific file names, sheet names, page numbers, or row ranges when answering from CRM data.
5. Be precise with numbers and data points.
6. Format your response as JSON with keys: "answer" (string), "citations" (list of objects with file, sheet, page, rows as applicable)"""


class GeminiLLM:
    """Wrapper around Google's Gemini API via the GenAI SDK.

    Attributes:
        model: The Gemini model identifier to use for generation.
        client: An authenticated ``genai.Client`` instance.
    """

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash-latest") -> None:
        """Initialise the Gemini client.

        Args:
            api_key: A valid Google AI API key.
            model: The model name to use (default ``gemini-2.5-flash``).
        """
        if not api_key:
            raise ValueError("A non-empty API key is required to initialise GeminiLLM.")
        self.model = model
        self.client = genai.Client(api_key=api_key)
        logger.info("GeminiLLM initialised with model=%s", self.model)

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
    ) -> str:
        """Generate a plain-text response from Gemini.

        Args:
            prompt: The user prompt / contents to send.
            system_instruction: Optional system-level instruction prepended
                to the conversation.

        Returns:
            The model's text response.

        Raises:
            RuntimeError: If the API call fails.
        """
        try:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            text = response.text
            logger.debug("Gemini response length: %d chars", len(text))
            return text
        except Exception as exc:
            logger.error("Gemini API call failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Gemini generation failed: {exc}") from exc

    def generate_json(
        self,
        prompt: str,
        system_instruction: str | None = None,
    ) -> dict:
        """Generate a response and parse it as JSON.

        If the model wraps its output in markdown code fences (````json … ````)
        they are stripped before parsing.  When JSON parsing fails the raw text
        is returned inside ``{"raw_response": "<text>"}``.

        Args:
            prompt: The user prompt / contents to send.
            system_instruction: Optional system-level instruction.

        Returns:
            A parsed JSON dictionary, or a fallback dict with the raw text.
        """
        raw_text = self.generate(prompt, system_instruction=system_instruction)

        # Strip optional markdown code fences.
        cleaned = raw_text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            parsed: dict = json.loads(cleaned)
            logger.debug("Successfully parsed JSON response.")
            return parsed
        except json.JSONDecodeError as exc:
            logger.warning(
                "Failed to parse LLM response as JSON (%s). Returning raw text.",
                exc,
            )
            return {"raw_response": raw_text}
