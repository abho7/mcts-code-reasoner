"""
Translates an ExecutionReport (raw test-case results) into a scalar
process reward for MCTS backpropagation, and into a verbal failure
summary for the self-reflection step.

Reward shaping matters here: a program that passes 8/10 test cases and
one that passes 0/10 with a syntax error are both "failures", but they
represent very different amounts of progress, and the search should be
able to tell them apart. Pass-rate is the base signal; a small structural
penalty is layered on for failure categories that indicate the draft
isn't just wrong but actively broken (crashes, timeouts), since those
often need a different kind of fix (a bug patch) than a wrong-output
failure on an edge case (which often just needs different logic, not
debugging).
"""

from __future__ import annotations

from reasoner.execution.sandbox import ExecutionReport, ExecutionStatus, TestCaseResult

# Penalty applied on top of (1 - pass_rate) for failure categories that
# indicate a structurally broken draft rather than a logically wrong one.
_STRUCTURAL_PENALTY = {
    ExecutionStatus.RUNTIME_ERROR: 0.15,
    ExecutionStatus.TIMEOUT: 0.10,
    ExecutionStatus.MEMORY_EXCEEDED: 0.10,
    ExecutionStatus.COMPILE_ERROR: 0.25,
    ExecutionStatus.WRONG_OUTPUT: 0.0,
    ExecutionStatus.SUCCESS: 0.0,
}


def compute_reward(report: ExecutionReport) -> float:
    """Reward in [0, 1]. 1.0 only when every test case passes."""
    if not report.results:
        return 0.0
    if report.all_passed:
        return 1.0

    pass_rate = report.pass_rate
    penalty = sum(_STRUCTURAL_PENALTY[r.status] for r in report.results if not r.passed) / len(report.results)
    return max(0.0, pass_rate - penalty)


def summarize_failure(report: ExecutionReport, max_stderr_chars: int = 800) -> str:
    """Human/LLM-readable summary of the first failure, for the
    self-reflection prompt. Deliberately reports only the *first*
    failing test case in detail -- dumping every failure at once tends
    to produce reflection prompts an LLM pattern-matches into a vague
    'fix everything' patch rather than a targeted one."""
    failure = report.first_failure
    if failure is None:
        return "All test cases passed."

    lines = [
        f"Test '{failure.test_id}' failed with status: {failure.status.value}",
        f"Wall time: {failure.wall_time_seconds:.3f}s",
    ]
    if failure.status == ExecutionStatus.TIMEOUT:
        lines.append("The program did not terminate within the time limit -- likely an infinite loop, "
                      "unbounded recursion, or an algorithm with too high a time complexity for the input size.")
    elif failure.status == ExecutionStatus.MEMORY_EXCEEDED:
        lines.append("The program exceeded the memory limit -- likely an unbounded data structure or "
                      "an algorithm with too high a space complexity.")
    elif failure.status == ExecutionStatus.RUNTIME_ERROR:
        stderr = failure.stderr.strip()
        if len(stderr) > max_stderr_chars:
            stderr = stderr[-max_stderr_chars:]  # tail, since the actual exception is at the end
        lines.append(f"Traceback (most recent call last, truncated to last {max_stderr_chars} chars):\n{stderr}")
    elif failure.status == ExecutionStatus.WRONG_OUTPUT:
        lines.append(f"Expected output did not match. Got:\n{failure.stdout.strip()[:500]}")

    lines.append(f"Overall pass rate on this draft: {report.pass_rate:.1%} ({sum(r.passed for r in report.results)}/{len(report.results)} test cases)")
    return "\n".join(lines)
