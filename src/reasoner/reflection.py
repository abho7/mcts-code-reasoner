"""
Constructs the "verbal self-reflection" prompt: takes a failing code
draft plus its execution failure summary and produces the system/user
prompt pair the LLM sees when generating a FIX_BUG action.

This is deliberately a separate module from execution/reward.py (which
produces the raw failure summary) and mcts/search.py (which decides
*when* to invoke reflection during the search) -- prompt construction is
the piece most likely to need iteration/tuning independent of the search
algorithm itself, so it's isolated here.
"""

from __future__ import annotations

from dataclasses import dataclass

SYSTEM_PROMPT = """You are an expert competitive programmer performing iterative self-correction.
You will be shown a Python program you previously wrote, the problem it's meant to solve, and the
exact execution failure it produced. Your job is to produce a corrected FULL program (not a diff,
not an explanation -- just the complete corrected Python source) that fixes the specific failure
shown, without introducing regressions on behavior that was already correct.

Rules:
- Output ONLY the corrected Python source code, no markdown fences, no commentary.
- Read stdin and write stdout in exactly the format the problem specifies.
- Fix the root cause shown in the failure, not just its symptom.
"""


@dataclass(frozen=True)
class ReflectionPrompt:
    system_prompt: str
    user_prompt: str


def build_reflection_prompt(
    problem_statement: str,
    previous_code: str,
    failure_summary: str,
    prior_attempts_summary: str | None = None,
) -> ReflectionPrompt:
    """
    prior_attempts_summary: optional short note on what earlier branches
    in the search tree already tried and failed at (e.g. "A previous
    attempt using recursion hit a timeout on large inputs"). This is
    what lets the search avoid re-exploring a dead branch's mistake in a
    sibling branch -- it's populated from the failure reasons stored on
    other nodes along the search tree, not just this node's own history.
    """
    sections = [
        f"## Problem\n{problem_statement.strip()}",
        f"## Previous attempt\n```python\n{previous_code.strip()}\n```",
        f"## Execution failure\n{failure_summary.strip()}",
    ]
    if prior_attempts_summary:
        sections.append(f"## Other approaches already tried and failed\n{prior_attempts_summary.strip()}")
    sections.append("## Task\nProduce the corrected full program.")

    return ReflectionPrompt(
        system_prompt=SYSTEM_PROMPT,
        user_prompt="\n\n".join(sections),
    )


def build_initial_generation_prompt(problem_statement: str) -> ReflectionPrompt:
    system_prompt = """You are an expert competitive programmer. Given a problem statement, produce a
complete, correct Python program that reads from stdin and writes to stdout exactly as specified.
Output ONLY the Python source code, no markdown fences, no commentary."""
    return ReflectionPrompt(
        system_prompt=system_prompt,
        user_prompt=f"## Problem\n{problem_statement.strip()}\n\n## Task\nProduce a complete solution.",
    )


def build_optimization_prompt(problem_statement: str, passing_code: str, target_note: str) -> ReflectionPrompt:
    """Used for the OPTIMIZE action on a draft that already passes all
    test cases but may not meet the problem's stated complexity bound
    (e.g. an O(n^2) solution on a problem with n up to 10^6)."""
    system_prompt = """You are an expert competitive programmer. You will be shown a program that
already passes all known test cases, but may be too slow or memory-heavy for the problem's actual
constraints. Produce a functionally equivalent but more efficient version. Output ONLY the Python
source code, no markdown fences, no commentary."""
    user_prompt = (
        f"## Problem\n{problem_statement.strip()}\n\n"
        f"## Current passing solution\n```python\n{passing_code.strip()}\n```\n\n"
        f"## Efficiency concern\n{target_note.strip()}\n\n"
        "## Task\nProduce a more efficient equivalent solution."
    )
    return ReflectionPrompt(system_prompt=system_prompt, user_prompt=user_prompt)
