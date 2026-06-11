"""
llm_router.py — Multi-Provider LLM Router

Responsibility:
    Route LLM generation requests to one of three providers:
      - Claude (Anthropic)  — primary
      - OpenAI (gpt-4o-mini) — secondary
      - Gemini (gemini-2.0-flash) — tertiary

    Supports:
      - Explicit provider selection ("claude", "openai", "gemini")
      - "auto" mode: tries providers in fallback chain order
      - Streaming and non-streaming responses
      - Graceful degradation — skips providers with missing keys

Architecture: Strategy Pattern
    Each provider implements a generate() strategy.
    The router selects and delegates — the agent never talks to
    an LLM client directly, making provider swaps transparent.

CV-aligned features:
    - Multi-provider routing across OpenAI / Claude / Gemini
    - 4-layer safety stack integration (Layer 4 = provider routing gate)
"""

import os
import sys
from typing import Iterator, Optional
from dataclasses import dataclass

from config import (
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY,
    GENERATION_MODEL, OPENAI_MODEL, GEMINI_MODEL,
    DEFAULT_PROVIDER, PROVIDER_FALLBACK_CHAIN,
)


# ── Provider Availability Check ──────────────────────────────────────────────

def _available_providers() -> list[str]:
    """Return providers that have API keys configured."""
    available = []
    if ANTHROPIC_API_KEY:
        available.append("claude")
    if OPENAI_API_KEY:
        available.append("openai")
    if GEMINI_API_KEY:
        available.append("gemini")
    return available


# ── Result Dataclass ──────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    """
    Unified response object from any provider.

    Fields:
        text          : Full generated text
        provider      : Which provider generated it ("claude"/"openai"/"gemini")
        model         : Specific model used
        input_tokens  : Input token count (if available)
        output_tokens : Output token count (if available)
        error         : Error message if generation failed
        success       : False if all providers failed
    """
    text: str = ""
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""
    success: bool = True


# ── Per-Provider Generators ───────────────────────────────────────────────────

def _stream_claude(messages: list[dict], system: str) -> Iterator[str]:
    """Stream tokens from Claude (Anthropic)."""
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    with client.messages.stream(
        model=GENERATION_MODEL,
        max_tokens=1024,
        system=system,
        messages=messages,
    ) as stream:
        for token in stream.text_stream:
            yield token


def _generate_claude(messages: list[dict], system: str) -> LLMResponse:
    """Non-streaming Claude generation."""
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=1024,
        system=system,
        messages=messages,
    )
    return LLMResponse(
        text=response.content[0].text,
        provider="claude",
        model=GENERATION_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


def _stream_openai(messages: list[dict], system: str) -> Iterator[str]:
    """Stream tokens from OpenAI."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    full_messages = [{"role": "system", "content": system}] + messages
    stream = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=full_messages,
        max_tokens=1024,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


def _generate_openai(messages: list[dict], system: str) -> LLMResponse:
    """Non-streaming OpenAI generation."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    full_messages = [{"role": "system", "content": system}] + messages
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=full_messages,
        max_tokens=1024,
    )
    return LLMResponse(
        text=response.choices[0].message.content,
        provider="openai",
        model=OPENAI_MODEL,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
    )


def _stream_gemini(messages: list[dict], system: str) -> Iterator[str]:
    """Stream tokens from Google Gemini."""
    from google import genai
    from google.genai import types as genai_types
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Convert messages to Gemini format
    contents = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append(genai_types.Content(
            role=role,
            parts=[genai_types.Part(text=m["content"])],
        ))

    response = client.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=contents,
        config=genai_types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=1024,
        ),
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text


def _generate_gemini(messages: list[dict], system: str) -> LLMResponse:
    """Non-streaming Gemini generation."""
    from google import genai
    from google.genai import types as genai_types
    client = genai.Client(api_key=GEMINI_API_KEY)

    contents = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append(genai_types.Content(
            role=role,
            parts=[genai_types.Part(text=m["content"])],
        ))

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=genai_types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=1024,
        ),
    )
    return LLMResponse(
        text=response.text,
        provider="gemini",
        model=GEMINI_MODEL,
    )


# ── Provider Dispatch Tables ──────────────────────────────────────────────────

_STREAMERS = {
    "claude": _stream_claude,
    "openai": _stream_openai,
    "gemini": _stream_gemini,
}

_GENERATORS = {
    "claude": _generate_claude,
    "openai": _generate_openai,
    "gemini": _generate_gemini,
}


# ── LLMRouter (Main Class) ────────────────────────────────────────────────────

class LLMRouter:
    """
    Multi-provider LLM router with automatic fallback.

    Usage (streaming):
        router = LLMRouter(provider="auto")
        for token in router.stream(messages, system):
            print(token, end="", flush=True)

    Usage (non-streaming):
        router = LLMRouter(provider="gemini")
        response = router.generate(messages, system)
        print(response.text)

    The router records which provider was actually used in
    `self.last_provider` for logging / cost tracking.
    """

    def __init__(self, provider: str = "auto"):
        """
        Args:
            provider : "auto", "claude", "openai", or "gemini"
        """
        self.requested_provider = provider
        self.last_provider: Optional[str] = None
        self._available = _available_providers()

        if not self._available:
            raise EnvironmentError(
                "No LLM provider keys configured. "
                "Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY."
            )

    def _resolve_chain(self) -> list[str]:
        """
        Determine the ordered list of providers to attempt.
        Returns a list starting with the requested provider (or fallback order).
        """
        if self.requested_provider == "auto":
            # Try in PROVIDER_FALLBACK_CHAIN order, skip unavailable ones
            return [p for p in PROVIDER_FALLBACK_CHAIN if p in self._available]

        if self.requested_provider in self._available:
            # Specific provider requested and available
            # Still add fallbacks in case it errors
            fallback = [p for p in PROVIDER_FALLBACK_CHAIN
                        if p != self.requested_provider and p in self._available]
            return [self.requested_provider] + fallback

        # Requested provider not available — use auto chain
        print(
            f"[LLMRouter] Warning: '{self.requested_provider}' not available "
            f"(missing API key). Falling back to auto chain.",
            file=sys.stderr,
        )
        return [p for p in PROVIDER_FALLBACK_CHAIN if p in self._available]

    def stream(
        self,
        messages: list[dict],
        system: str = "",
    ) -> Iterator[tuple[str, str]]:
        """
        Stream tokens from the best available provider.

        Yields:
            Tuple of (token_text, provider_name)
            Yields ("__PROVIDER__", provider_name) as first item to signal which was chosen.

        Tries providers in fallback order. On error, logs and tries next.
        """
        chain = self._resolve_chain()
        if not chain:
            raise RuntimeError("No providers available with valid API keys.")

        for provider in chain:
            try:
                self.last_provider = provider
                yield ("__PROVIDER__", provider)
                for token in _STREAMERS[provider](messages, system):
                    yield (token, provider)
                return  # Success — stop after first working provider
            except Exception as e:
                print(
                    f"[LLMRouter] Provider '{provider}' failed: {e}. "
                    f"Trying next in chain...",
                    file=sys.stderr,
                )
                continue

        raise RuntimeError(
            f"All providers failed. Chain attempted: {chain}"
        )

    def generate(
        self,
        messages: list[dict],
        system: str = "",
    ) -> LLMResponse:
        """
        Non-streaming generation with automatic fallback.

        Returns:
            LLMResponse with text, provider, model, and token counts.
            On failure of all providers, returns LLMResponse(success=False).
        """
        chain = self._resolve_chain()
        if not chain:
            return LLMResponse(
                success=False,
                error="No providers available with valid API keys.",
            )

        last_error = ""
        for provider in chain:
            try:
                self.last_provider = provider
                response = _GENERATORS[provider](messages, system)
                return response
            except Exception as e:
                last_error = str(e)
                print(
                    f"[LLMRouter] Provider '{provider}' failed: {e}. "
                    f"Trying next...",
                    file=sys.stderr,
                )
                continue

        return LLMResponse(
            success=False,
            error=f"All providers failed. Last error: {last_error}",
        )

    @property
    def available_providers(self) -> list[str]:
        """Return list of providers with configured API keys."""
        return self._available.copy()


# ── Convenience Factory ───────────────────────────────────────────────────────

def get_router(provider: str = None) -> LLMRouter:
    """
    Factory function to get an LLMRouter instance.

    Args:
        provider : Override provider. None = use DEFAULT_PROVIDER from config.
    """
    return LLMRouter(provider=provider or DEFAULT_PROVIDER)


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing LLMRouter...\n")

    available = _available_providers()
    print(f"Available providers: {available}")

    if not available:
        print("[ERROR] No API keys configured. Set keys in .env file.")
        sys.exit(1)

    router = get_router("auto")
    print(f"Router created | auto chain: {router._resolve_chain()}\n")

    # Test non-streaming generation
    print("=" * 50)
    print("Test: Non-streaming generate()")
    print("=" * 50)
    test_messages = [{"role": "user", "content": "Say 'Hello from {provider}' in exactly 5 words."}]
    response = router.generate(test_messages, system="You are a helpful assistant.")
    print(f"Provider : {response.provider}")
    print(f"Model    : {response.model}")
    print(f"Text     : {response.text}")
    print(f"Tokens   : {response.input_tokens} in / {response.output_tokens} out")

    print("\n[OK] LLMRouter ready!")
