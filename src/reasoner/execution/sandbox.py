"""
Isolated execution environment for benchmarking LLM-generated code against
test cases, under hard resource walls.

Security model
---------------
This is a *process-level* sandbox, not a container or VM. It is
appropriate for benchmarking algorithmic code (competitive-programming
style solutions: pure computation, stdin/stdout, no filesystem or network
need) against a resource budget. It is NOT a substitute for a real
isolation boundary (gVisor, Firecracker, a Docker container with dropped
capabilities, a seccomp profile) if the code you're executing is
untrusted in a stronger sense than "might have bugs" -- e.g. if it might
try to read secrets, reach the network, or fork-bomb the host. The
resource limits below stop runaway CPU/memory/wall-time, not a
deliberately malicious escape attempt.

Three independent walls are enforced, because any one of them can fail
to catch a given failure mode on its own:
  1. RLIMIT_AS (address space) -- catches unbounded memory allocation.
  2. RLIMIT_CPU (CPU seconds) -- catches CPU-bound infinite loops even if
     the OS scheduler is under load and wall-clock time lags CPU time.
  3. An outer asyncio.wait_for wall-clock timeout -- catches everything
     else (a hung syscall, I/O wait, a CPU limit set too generously),
     and is what actually kills the process from the parent side.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum

try:
    import resource
    _HAS_RESOURCE_MODULE = True
except ImportError:
    # `resource` is POSIX-only and doesn't exist on Windows at all. On
    # Windows, memory (RLIMIT_AS) and CPU-time (RLIMIT_CPU) walls are
    # simply unavailable -- the wall-clock asyncio.wait_for timeout below
    # is still enforced and is what actually kills a hung process, but a
    # script that allocates memory without looping (and so never blocks
    # long enough to hit the wall-clock timeout) will not be stopped on
    # Windows the way it would be on Linux/macOS. Run in a container or
    # WSL for the full set of resource walls in production.
    _HAS_RESOURCE_MODULE = False


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    WRONG_OUTPUT = "wrong_output"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    MEMORY_EXCEEDED = "memory_exceeded"
    COMPILE_ERROR = "compile_error"  # reserved for compiled-language executors


@dataclass
class TestCaseResult:
    test_id: str
    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    wall_time_seconds: float = 0.0
    exit_code: int | None = None

    @property
    def passed(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS


@dataclass
class ExecutionReport:
    """Aggregate result of running one code draft against every test case."""
    results: list[TestCaseResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.passed for r in self.results) / len(self.results)

    @property
    def all_passed(self) -> bool:
        return len(self.results) > 0 and all(r.passed for r in self.results)

    @property
    def first_failure(self) -> TestCaseResult | None:
        for r in self.results:
            if not r.passed:
                return r
        return None


@dataclass(frozen=True)
class SandboxLimits:
    timeout_seconds: float = 2.0
    memory_limit_mb: int = 256
    cpu_time_limit_seconds: float = 3.0  # slightly above wall timeout; wall clock is the real backstop


def _preexec_fn(memory_limit_mb: int, cpu_time_limit_seconds: float):
    """Runs inside the forked child, before exec. Sets hard resource
    ceilings on the child process itself -- this is what makes the limits
    apply even if the parent's wait_for() were somehow bypassed.

    Returns None on Windows (where neither the `resource` module nor
    subprocess's `preexec_fn` argument exist at all), signaling callers
    to skip passing preexec_fn to subprocess creation entirely."""
    if not _HAS_RESOURCE_MODULE:
        return None

    def _set_limits():
        mem_bytes = memory_limit_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, resource.error):
            pass  # some platforms don't support RLIMIT_AS; CPU + wall-clock still apply
        cpu_secs = max(1, int(cpu_time_limit_seconds))
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_secs, cpu_secs))
        except (ValueError, resource.error):
            pass
    return _set_limits


class CodeExecutor:
    """Runs a Python code draft against a list of (stdin, expected_stdout)
    test cases, each under the configured resource walls, concurrently."""

    def __init__(self, limits: SandboxLimits | None = None):
        self.limits = limits or SandboxLimits()

    async def run_test_case(self, code: str, test_id: str, stdin: str, expected_stdout: str) -> TestCaseResult:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            script_path = f.name

        start = time.monotonic()
        proc: asyncio.subprocess.Process | None = None
        try:
            subprocess_kwargs = dict(
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            preexec = _preexec_fn(self.limits.memory_limit_mb, self.limits.cpu_time_limit_seconds)
            if preexec is not None:
                subprocess_kwargs["preexec_fn"] = preexec  # POSIX only; omitted entirely on Windows

            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-I", "-S", script_path,  # -I isolated, -S no site imports: reduces attack surface
                **subprocess_kwargs,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(stdin.encode()), timeout=self.limits.timeout_seconds
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return TestCaseResult(
                    test_id=test_id,
                    status=ExecutionStatus.TIMEOUT,
                    wall_time_seconds=time.monotonic() - start,
                    exit_code=None,
                )

            elapsed = time.monotonic() - start
            stdout = stdout_b.decode(errors="replace")
            stderr = stderr_b.decode(errors="replace")
            exit_code = proc.returncode

            if exit_code == -9 or "MemoryError" in stderr:
                return TestCaseResult(
                    test_id=test_id, status=ExecutionStatus.MEMORY_EXCEEDED,
                    stdout=stdout, stderr=stderr, wall_time_seconds=elapsed, exit_code=exit_code,
                )
            if exit_code != 0:
                return TestCaseResult(
                    test_id=test_id, status=ExecutionStatus.RUNTIME_ERROR,
                    stdout=stdout, stderr=stderr, wall_time_seconds=elapsed, exit_code=exit_code,
                )
            if stdout.strip() != expected_stdout.strip():
                return TestCaseResult(
                    test_id=test_id, status=ExecutionStatus.WRONG_OUTPUT,
                    stdout=stdout, stderr=stderr, wall_time_seconds=elapsed, exit_code=exit_code,
                )
            return TestCaseResult(
                test_id=test_id, status=ExecutionStatus.SUCCESS,
                stdout=stdout, stderr=stderr, wall_time_seconds=elapsed, exit_code=exit_code,
            )
        finally:
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
            try:
                os.unlink(script_path)
            except OSError:
                pass

    async def run(self, code: str, test_cases: list[dict]) -> ExecutionReport:
        """test_cases: list of {"id": str, "stdin": str, "expected_stdout": str}.
        Runs all test cases concurrently -- this is the parallelization
        point referenced in the roadmap for reducing per-node evaluation
        latency during MCTS expansion."""
        tasks = [
            self.run_test_case(code, tc["id"], tc.get("stdin", ""), tc["expected_stdout"])
            for tc in test_cases
        ]
        results = await asyncio.gather(*tasks)
        return ExecutionReport(results=list(results))
