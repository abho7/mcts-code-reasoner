"""
Strict configuration schemas for every tunable in the system. Pydantic
validates types and bounds at construction time (a negative timeout, an
exploration constant of the wrong type, a memory limit given in bytes
instead of MB by mistake) fails immediately and legibly, rather than
producing a confusing failure three layers down during search.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LLMBackend(str, Enum):
    OLLAMA = "ollama"
    VLLM = "vllm"
    FAKE = "fake"  # test double, see reasoner.llm.client.FakeLLMClient


class LLMConfig(BaseModel):
    backend: LLMBackend = LLMBackend.OLLAMA
    model: str = Field(default="qwen2.5-coder:32b", description="Model identifier passed to the backend server.")
    base_url: str = Field(default="http://localhost:11434", description="Ollama or vLLM server URL.")
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_tokens_per_call: int = Field(default=1536, gt=0, le=32768)


class SandboxConfig(BaseModel):
    timeout_seconds: float = Field(default=2.0, gt=0, le=30.0, description="Wall-clock kill timeout per test case.")
    memory_limit_mb: int = Field(default=256, gt=0, le=8192)
    cpu_time_limit_seconds: float = Field(default=3.0, gt=0, le=60.0)
    max_concurrent_executions: int = Field(
        default=8, gt=0, le=256,
        description="Caps how many sandboxed subprocesses run at once, across all MCTS nodes being evaluated in parallel.",
    )

    @model_validator(mode="after")
    def _cpu_limit_should_exceed_wall_timeout(self) -> "SandboxConfig":
        if self.cpu_time_limit_seconds < self.timeout_seconds:
            raise ValueError(
                "cpu_time_limit_seconds should be >= timeout_seconds -- the wall-clock timeout is meant "
                "to be the primary backstop; a tighter CPU limit would trigger first and mask which wall "
                "actually caught the runaway process."
            )
        return self


class MCTSConfig(BaseModel):
    exploration_constant: float = Field(default=1.41421356, gt=0, description="UCB1 'c' term. sqrt(2) is the standard default.")
    token_penalty_weight: float = Field(
        default=0.3, ge=0,
        description="'lambda' in the UCB1_LLM budget-pressure term. 0 recovers standard UCB1.",
    )
    total_token_budget: int = Field(default=200_000, gt=0, description="Hard ceiling on total tokens spent across one search episode.")
    max_iterations: int = Field(default=64, gt=0, description="Hard ceiling on select/expand/simulate/backprop cycles, independent of token budget.")
    max_tree_depth: int = Field(default=12, gt=0)
    reward_success_threshold: float = Field(
        default=1.0, ge=0, le=1.0,
        description="Search stops early once a node's reward meets or exceeds this (default: only a full pass ends the search).",
    )
    simulations_per_expansion: int = Field(default=1, ge=1, description="How many independent rollouts to average per newly expanded node.")


class RunConfig(BaseModel):
    """Top-level config for a single problem-solving episode."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    mcts: MCTSConfig = Field(default_factory=MCTSConfig)
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")

    model_config = ConfigDict(extra="forbid")  # catch typo'd config keys instead of silently ignoring them
