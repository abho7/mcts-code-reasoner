import asyncio

from reasoner.execution.sandbox import CodeExecutor, ExecutionStatus, SandboxLimits


def _run(coro):
    return asyncio.run(coro)


def test_correct_solution_passes():
    executor = CodeExecutor(SandboxLimits(timeout_seconds=1.5))
    code = "a, b = map(int, input().split()); print(a + b)"
    report = _run(executor.run(code, [
        {"id": "t1", "stdin": "2 3", "expected_stdout": "5"},
        {"id": "t2", "stdin": "10 20", "expected_stdout": "30"},
    ]))
    assert report.all_passed
    assert report.pass_rate == 1.0


def test_wrong_output_detected():
    executor = CodeExecutor(SandboxLimits(timeout_seconds=1.5))
    code = "a, b = map(int, input().split()); print(a - b)"
    report = _run(executor.run(code, [{"id": "t1", "stdin": "2 3", "expected_stdout": "5"}]))
    assert not report.all_passed
    assert report.results[0].status == ExecutionStatus.WRONG_OUTPUT


def test_infinite_loop_is_killed_by_timeout():
    executor = CodeExecutor(SandboxLimits(timeout_seconds=1.0))
    report = _run(executor.run("while True: pass", [{"id": "t1", "stdin": "", "expected_stdout": ""}]))
    assert report.results[0].status == ExecutionStatus.TIMEOUT
    assert report.results[0].wall_time_seconds < 3.0  # actually got killed, didn't hang


def test_memory_bomb_does_not_silently_succeed():
    """On POSIX (Linux/macOS), this is caught cleanly by RLIMIT_AS as
    MEMORY_EXCEEDED. On Windows, `resource` doesn't exist at all, so the
    guarantee is weaker: the wall-clock timeout is still the backstop,
    and the allocation may instead surface as a real Python MemoryError
    (RUNTIME_ERROR) or a TIMEOUT if the allocation itself is slow. What
    must never happen on any platform is silently succeeding, which is
    what this test actually guards -- see sandbox.py's module docstring
    for the full platform caveat."""
    executor = CodeExecutor(SandboxLimits(timeout_seconds=2.0, memory_limit_mb=100))
    report = _run(executor.run("x = [0] * (10**9)", [{"id": "t1", "stdin": "", "expected_stdout": ""}]))
    assert report.results[0].status in (
        ExecutionStatus.MEMORY_EXCEEDED,
        ExecutionStatus.RUNTIME_ERROR,
        ExecutionStatus.TIMEOUT,
    )


def test_runtime_error_captures_traceback():
    executor = CodeExecutor(SandboxLimits(timeout_seconds=1.5))
    report = _run(executor.run("x = [1, 2, 3]\nprint(x[10])", [{"id": "t1", "stdin": "", "expected_stdout": ""}]))
    assert report.results[0].status == ExecutionStatus.RUNTIME_ERROR
    assert "IndexError" in report.results[0].stderr


def test_concurrent_execution_is_actually_parallel():
    import time
    executor = CodeExecutor(SandboxLimits(timeout_seconds=1.5))
    code = "a, b = map(int, input().split()); print(a + b)"
    test_cases = [{"id": f"t{i}", "stdin": "1 1", "expected_stdout": "2"} for i in range(10)]
    start = time.monotonic()
    report = _run(executor.run(code, test_cases))
    elapsed = time.monotonic() - start
    assert report.all_passed
    assert elapsed < 3.0  # 10 sequential runs would take much longer if not actually concurrent
