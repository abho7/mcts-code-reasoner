#!/usr/bin/env python3
"""
Demo entry point. Two modes:

    python scripts/run_example.py --fake
        Runs against FakeLLMClient with a scripted correct-on-first-try
        response -- a smoke test that exercises the full pipeline
        (config validation, sandbox, MCTS, reward) with no model server
        required. This is what you should run first to confirm the
        install is sound before pointing it at a real model.

    python scripts/run_example.py --backend ollama --model qwen2.5-coder:32b
        Runs against a live Ollama server (must already be running with
        the model pulled: `ollama pull qwen2.5-coder:32b`).

Requires `pip install -r requirements.txt` first (this script imports
reasoner.config, which has a hard pydantic dependency by design -- see
the module docstring in src/reasoner/orchestrator.py for why).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MCTS code-reasoning engine against a problem.")
    parser.add_argument("--problem", default="problems/two_sum_indices.json", help="Path to a problem JSON file.")
    parser.add_argument("--backend", choices=["ollama", "vllm", "fake"], default="fake")
    parser.add_argument("--model", default="qwen2.5-coder:32b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--token-budget", type=int, default=200_000)
    parser.add_argument("--max-iterations", type=int, default=32)
    args = parser.parse_args()

    from reasoner.config import LLMBackend, LLMConfig, MCTSConfig, RunConfig, SandboxConfig
    from reasoner.orchestrator import solve_problem

    problem_path = Path(args.problem)
    if not problem_path.is_absolute():
        problem_path = Path(__file__).parent.parent / problem_path
    problem = json.loads(problem_path.read_text())

    config = RunConfig(
        llm=LLMConfig(backend=LLMBackend(args.backend), model=args.model, base_url=args.base_url),
        sandbox=SandboxConfig(),
        mcts=MCTSConfig(total_token_budget=args.token_budget, max_iterations=args.max_iterations),
    )

    fake_responses = None
    if args.backend == "fake":
        # A trivially correct solution to two_sum_indices, for the smoke test.
        fake_responses = [
            "arr = list(map(int, input().split()))\n"
            "target = int(input())\n"
            "seen = {}\n"
            "for i, v in enumerate(arr):\n"
            "    need = target - v\n"
            "    if need in seen:\n"
            "        print(seen[need], i)\n"
            "        break\n"
            "    seen[v] = i\n"
        ]

    result = asyncio.run(
        solve_problem(
            config,
            problem_statement=problem["statement"],
            test_cases=problem["test_cases"],
            fake_responses=fake_responses,
        )
    )

    print()
    print("=" * 60)
    print(f"Terminal (all tests passed): {result.is_terminal}")
    print(f"Reward (q_value):            {result.q_value:.3f}")
    print(f"Search depth:                {result.depth}")
    print("=" * 60)
    print(result.state_code)


if __name__ == "__main__":
    main()
