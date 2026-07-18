import asyncio

from reasoner.execution.sandbox import CodeExecutor, SandboxLimits
from reasoner.llm.client import FakeLLMClient
from reasoner.mcts.search import MCTSSearch


def _run(coro):
    return asyncio.run(coro)


def test_search_self_corrects_a_buggy_first_draft():
    """This is the core claim of the whole project: a failing draft's
    execution failure gets fed back into the LLM, and the search
    converges on a correct solution. Verified end-to-end with a scripted
    LLM standing in for a real model."""
    buggy = "a, b = map(int, input().split())\nprint(a - b)  # bug"
    fixed = "a, b = map(int, input().split())\nprint(a + b)"

    fake_llm = FakeLLMClient(scripted_responses=[buggy, fixed])
    executor = CodeExecutor(SandboxLimits(timeout_seconds=1.0))

    search = MCTSSearch(
        llm_client=fake_llm,
        executor=executor,
        problem_statement="Read two integers a and b from stdin, print a + b.",
        test_cases=[
            {"id": "t1", "stdin": "2 3", "expected_stdout": "5"},
            {"id": "t2", "stdin": "10 20", "expected_stdout": "30"},
        ],
        max_iterations=10,
        total_token_budget=50_000,
    )

    result = _run(search.run())

    assert result.is_terminal
    assert result.q_value == 1.0
    assert "a + b" in result.state_code
    assert search.iterations_run == 2


def test_search_converges_immediately_when_first_draft_is_correct():
    correct = "a, b = map(int, input().split())\nprint(a + b)"
    fake_llm = FakeLLMClient(scripted_responses=[correct])
    executor = CodeExecutor(SandboxLimits(timeout_seconds=1.0))

    search = MCTSSearch(
        llm_client=fake_llm,
        executor=executor,
        problem_statement="Read two integers a and b from stdin, print a + b.",
        test_cases=[{"id": "t1", "stdin": "2 3", "expected_stdout": "5"}],
        max_iterations=10,
    )

    result = _run(search.run())
    assert result.is_terminal
    assert search.iterations_run == 1


def test_search_returns_best_effort_when_budget_or_script_runs_out():
    """If the model never gets it right, the search should still return
    its best attempt so far rather than crashing -- important for
    production use where you always want *a* submission, even a partial
    one, rather than an unhandled exception."""
    always_wrong = "print('definitely wrong')"
    fake_llm = FakeLLMClient(scripted_responses=[always_wrong] * 3)
    executor = CodeExecutor(SandboxLimits(timeout_seconds=1.0))

    search = MCTSSearch(
        llm_client=fake_llm,
        executor=executor,
        problem_statement="Read two integers a and b from stdin, print a + b.",
        test_cases=[{"id": "t1", "stdin": "2 3", "expected_stdout": "5"}],
        max_iterations=3,
    )

    result = _run(search.run())
    assert not result.is_terminal
    assert result.q_value < 1.0


def test_tokens_spent_is_tracked_across_iterations():
    buggy = "print('x')"
    fixed = "a, b = map(int, input().split())\nprint(a + b)"
    fake_llm = FakeLLMClient(scripted_responses=[buggy, fixed])
    executor = CodeExecutor(SandboxLimits(timeout_seconds=1.0))

    search = MCTSSearch(
        llm_client=fake_llm,
        executor=executor,
        problem_statement="Read two integers a and b from stdin, print a + b.",
        test_cases=[{"id": "t1", "stdin": "2 3", "expected_stdout": "5"}],
        max_iterations=10,
    )
    _run(search.run())
    assert search.tokens_spent > 0
    assert search.tokens_remaining == search.total_token_budget - search.tokens_spent
