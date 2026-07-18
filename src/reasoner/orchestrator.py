"""
Top-level entry point: takes a validated RunConfig + a problem, builds
the concrete LLMClient/CodeExecutor from that config, and drives an
MCTSSearch episode. This is the layer that has a hard dependency on
pydantic (via config.py) -- the search algorithm itself (mcts/search.py)
deliberately does not, so it stays testable without pydantic installed.
"""

from __future__ import annotations

import logging

from reasoner.config import LLMBackend, RunConfig
from reasoner.execution.sandbox import CodeExecutor, SandboxLimits
from reasoner.llm.client import FakeLLMClient, LLMClient, OllamaClient, VLLMClient
from reasoner.logging_utils import configure_logging
from reasoner.mcts.node import MCTSNode
from reasoner.mcts.search import MCTSSearch

logger = logging.getLogger(__name__)


def _build_llm_client(config: RunConfig, fake_responses: list[str] | None = None) -> LLMClient:
    if config.llm.backend == LLMBackend.OLLAMA:
        return OllamaClient(model=config.llm.model, base_url=config.llm.base_url, timeout_seconds=config.llm.timeout_seconds)
    if config.llm.backend == LLMBackend.VLLM:
        return VLLMClient(model=config.llm.model, base_url=config.llm.base_url, timeout_seconds=config.llm.timeout_seconds)
    if config.llm.backend == LLMBackend.FAKE:
        if not fake_responses:
            raise ValueError("LLMBackend.FAKE requires fake_responses to be provided (for tests/demos).")
        return FakeLLMClient(scripted_responses=fake_responses)
    raise ValueError(f"Unknown LLM backend: {config.llm.backend}")


async def solve_problem(
    config: RunConfig,
    problem_statement: str,
    test_cases: list[dict],
    *,
    fake_responses: list[str] | None = None,
) -> MCTSNode:
    """Runs one full MCTS episode against a problem and returns the best
    node found. Raises if the search never manages to expand a single
    node (malformed problem/test_cases input)."""
    configure_logging(config.log_level)
    logger.info("Starting search | backend=%s model=%s budget=%d tokens", config.llm.backend, config.llm.model, config.mcts.total_token_budget)

    llm_client = _build_llm_client(config, fake_responses=fake_responses)
    executor = CodeExecutor(
        SandboxLimits(
            timeout_seconds=config.sandbox.timeout_seconds,
            memory_limit_mb=config.sandbox.memory_limit_mb,
            cpu_time_limit_seconds=config.sandbox.cpu_time_limit_seconds,
        )
    )

    search = MCTSSearch(
        llm_client=llm_client,
        executor=executor,
        problem_statement=problem_statement,
        test_cases=test_cases,
        exploration_constant=config.mcts.exploration_constant,
        token_penalty_weight=config.mcts.token_penalty_weight,
        total_token_budget=config.mcts.total_token_budget,
        max_iterations=config.mcts.max_iterations,
        max_tree_depth=config.mcts.max_tree_depth,
        reward_success_threshold=config.mcts.reward_success_threshold,
    )

    result = await search.run()

    logger.info(
        "Search finished | terminal=%s q_value=%.3f iterations=%d tokens_spent=%d",
        result.is_terminal, result.q_value, search.iterations_run, search.tokens_spent,
    )
    return result
