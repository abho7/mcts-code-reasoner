"""
LLM interface the MCTS search drives against. Abstracted behind
`LLMClient` so the search logic (node.py, search.py) never depends on a
specific backend -- swap Ollama for vLLM, or for a test double, without
touching the search code.

Note on testability: this module talks to a network service (a local
Ollama or vLLM server) and cannot be exercised in an environment without
one running. `FakeLLMClient` at the bottom is not a toy -- it's what the
integration tests in tests/test_search_integration.py drive the real
MCTS loop against, so that the *search logic* is verified even where the
*model backend* can't be.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient(ABC):
    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 1024) -> LLMResponse:
        """Single-turn completion. The search layer builds full prompts
        (including any self-reflection context) before calling this --
        this interface deliberately doesn't manage conversation state,
        since MCTS branches need independent prompt construction per
        node, not a shared chat history."""
        raise NotImplementedError


class OllamaClient(LLMClient):
    """Talks to a local Ollama server's /api/generate endpoint. Requires
    `ollama serve` running and the target model pulled (e.g.
    `ollama pull qwen2.5-coder:32b`) -- this client does not manage the
    server lifecycle or model download."""

    def __init__(self, model: str = "qwen2.5-coder:32b", base_url: str = "http://localhost:11434", timeout_seconds: float = 120.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def complete(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 1024) -> LLMResponse:
        import httpx  # imported lazily: only needed if you actually run against a live Ollama server

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "system": system_prompt,
                    "prompt": user_prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return LLMResponse(
                text=data.get("response", ""),
                # Ollama reports these as prompt_eval_count / eval_count when available;
                # fall back to 0 rather than raising, since not every server build sets them.
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
            )


class VLLMClient(LLMClient):
    """Talks to a vLLM OpenAI-compatible server's /v1/chat/completions
    endpoint. Requires vLLM already serving the target model
    (`vllm serve Qwen/Qwen2.5-Coder-32B-Instruct`)."""

    def __init__(self, model: str = "Qwen/Qwen2.5-Coder-32B-Instruct", base_url: str = "http://localhost:8000/v1", timeout_seconds: float = 120.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def complete(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 1024) -> LLMResponse:
        import httpx  # imported lazily: only needed if you actually run against a live vLLM server

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": max_tokens,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage", {})
            return LLMResponse(
                text=data["choices"][0]["message"]["content"],
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            )


class FakeLLMClient(LLMClient):
    """Deterministic test double. Takes a script (an ordered list of
    responses) and returns them in sequence, ignoring the actual prompt
    content -- lets integration tests drive the real MCTS/executor/reward
    loop against known outputs (e.g. "buggy draft, then a fixed draft
    after seeing the traceback") without a live model."""

    def __init__(self, scripted_responses: list[str]):
        self._responses = list(scripted_responses)
        self._call_count = 0

    async def complete(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 1024) -> LLMResponse:
        if self._call_count >= len(self._responses):
            raise RuntimeError(
                f"FakeLLMClient exhausted its script after {self._call_count} calls -- "
                "add more scripted responses if the test needs more search iterations."
            )
        text = self._responses[self._call_count]
        self._call_count += 1
        # Fake but deterministic token accounting so budget-pressure logic is still exercisable in tests.
        return LLMResponse(text=text, prompt_tokens=len(user_prompt) // 4, completion_tokens=len(text) // 4)
