#!/usr/bin/env python3
"""
Equal-budget evaluation on HumanEval: greedy, best-of-N, sequential repair,
MCTS with standard UCB1 (lambda=0), and MCTS with the budget-aware UCB1.

Protocol (no test leakage):
  * Each HumanEval check() is split into its individual assert statements.
  * The first half (at least one) are VISIBLE tests: the only feedback any
    method sees during search, and the only thing early stopping looks at.
  * The returned program is then scored once on the FULL original check()
    in a fresh sandbox. That full-test result is what pass@1 reports.

Every method gets the same per-problem token budget (the binding
constraint, so the budget-aware penalty actually has something to act on)
plus a generous call cap, and every method stops as soon as a draft passes
the visible tests. A method may finish the call that crosses the budget.

Usage:
  # smoke test, no model needed (replays canonical solutions; expect 100%)
  python scripts/eval_humaneval.py --backend fake --limit 10

  # real run against a local Ollama model
  ollama pull qwen2.5-coder:7b
  python scripts/eval_humaneval.py --backend ollama --model qwen2.5-coder:7b

  # any OpenAI-compatible server (vLLM, LM Studio, llama.cpp server)
  python scripts/eval_humaneval.py --backend openai --base-url http://localhost:8000/v1 --model <name>

Results are appended to results/<run>.jsonl (resumable: finished
problem/method pairs are skipped on restart). A summary CSV and a LaTeX
table body are written at the end.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import gzip
import json
import math
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reasoner.execution.reward import compute_reward  # noqa: E402
from reasoner.execution.sandbox import CodeExecutor, ExecutionReport, SandboxLimits  # noqa: E402
from reasoner.llm.client import LLMClient, LLMResponse  # noqa: E402
from reasoner.mcts.search import MCTSSearch, _extract_code  # noqa: E402
from reasoner.reflection import build_initial_generation_prompt  # noqa: E402

METHODS = ["greedy", "best_of_n", "sequential_repair", "mcts_ucb1", "mcts_budget"]
LABELS = {
    "greedy": "Greedy (single call)",
    "best_of_n": "Best-of-N sampling",
    "sequential_repair": "Sequential repair ($k{=}1$)",
    "mcts_ucb1": "MCTS, standard UCB1 ($\\lambda{=}0$)",
    "mcts_budget": "MCTS, budget-aware UCB1 (ours)",
}


# --------------------------------------------------------------------- LLM
class HTTPClient(LLMClient):
    """Dependency-free client (urllib in a thread) for Ollama or any
    OpenAI-compatible server, with an explicit sampling temperature."""

    def __init__(self, backend: str, model: str, base_url: str, temperature: float, timeout: float = 300.0):
        self.backend, self.model, self.base_url = backend, model, base_url.rstrip("/")
        self.temperature, self.timeout = temperature, timeout
        self.calls = 0

    def _post(self, url: str, payload: dict) -> dict:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    async def complete(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 1024) -> LLMResponse:
        self.calls += 1
        if self.backend == "ollama":
            data = await asyncio.to_thread(self._post, f"{self.base_url}/api/generate", {
                "model": self.model, "system": system_prompt, "prompt": user_prompt, "stream": False,
                "options": {"num_predict": max_tokens, "temperature": self.temperature},
            })
            return LLMResponse(data.get("response", ""), data.get("prompt_eval_count", 0), data.get("eval_count", 0))
        data = await asyncio.to_thread(self._post, f"{self.base_url}/chat/completions", {
            "model": self.model, "max_tokens": max_tokens, "temperature": self.temperature,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        })
        u = data.get("usage", {})
        return LLMResponse(data["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0))


class CanonicalFakeClient(LLMClient):
    """Smoke-test double: always answers with the canonical solution."""

    def __init__(self, solution: str):
        self.solution, self.calls = solution, 0

    async def complete(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 1024) -> LLMResponse:
        self.calls += 1
        return LLMResponse(self.solution, len(user_prompt) // 4, len(self.solution) // 4)


# -------------------------------------------------------------- problems
@dataclass
class Problem:
    task_id: str
    prompt: str
    entry_point: str
    check_src: str        # full original test module (defines check)
    visible: list[str]    # individual assert statements used as feedback
    canonical: str


def load_humaneval(path: Path) -> list[Problem]:
    opener = gzip.open if path.suffix == ".gz" else open
    out = []
    with opener(path, "rt") as f:
        for line in f:
            d = json.loads(line)
            asserts = []
            for node in ast.walk(ast.parse(d["test"])):
                if isinstance(node, ast.FunctionDef) and node.name == "check":
                    asserts = [ast.unparse(s) for s in node.body if isinstance(s, ast.Assert)]
            visible = asserts[: max(1, len(asserts) // 2)] if asserts else []
            out.append(Problem(d["task_id"], d["prompt"], d["entry_point"], d["test"], visible,
                               d["prompt"] + d["canonical_solution"]))
    return out


def assemble(problem: Problem, code: str) -> str:
    """Model output -> runnable module. If the model returned only a body
    or dropped the signature, fall back to prompt + output."""
    code = _extract_code(code)
    if f"def {problem.entry_point}" not in code:
        code = problem.prompt + code
    return code


class FunctionTestExecutor:
    """Adapts HumanEval-style assert tests to the stdin/stdout CodeExecutor:
    each test becomes its own program (candidate + one assert + print OK)."""

    def __init__(self, inner: CodeExecutor, problem: Problem):
        self.inner, self.problem = inner, problem

    async def run(self, code: str, test_cases: list[dict]) -> ExecutionReport:
        prog = assemble(self.problem, code)
        tasks = [
            self.inner.run_test_case(
                f"{prog}\n\ncandidate = {self.problem.entry_point}\n{tc['assert_src']}\nprint('OK')\n",
                tc["id"], "", "OK")
            for tc in test_cases
        ]
        return ExecutionReport(results=list(await asyncio.gather(*tasks)))


def visible_cases(problem: Problem) -> list[dict]:
    if problem.visible:
        return [{"id": f"visible_{i}", "assert_src": a, "expected_stdout": "OK"} for i, a in enumerate(problem.visible)]
    # check() with no top-level asserts: use the whole thing as one visible test
    return [{"id": "visible_0", "assert_src": f"{problem.check_src}\ncheck(candidate)", "expected_stdout": "OK"}]


async def full_test_pass(executor: CodeExecutor, problem: Problem, code: str) -> bool:
    prog = f"{assemble(problem, code)}\n\n{problem.check_src}\ncheck({problem.entry_point})\nprint('OK')\n"
    res = await executor.run_test_case(prog, "full", "", "OK")
    return res.passed


# --------------------------------------------------------------- methods
async def run_method(method: str, problem: Problem, client_factory, executor: CodeExecutor, args) -> dict:
    fexec = FunctionTestExecutor(executor, problem)
    cases = visible_cases(problem)

    if method == "greedy":
        client = client_factory(0.0)
        p = build_initial_generation_prompt(problem.prompt, task_mode="function")
        r = await client.complete(p.system_prompt, p.user_prompt, max_tokens=args.max_tokens_per_call)
        code, tokens, calls = r.text, r.total_tokens, 1
        vis = (await fexec.run(code, cases)).all_passed

    elif method == "best_of_n":
        client = client_factory(args.temperature)
        p = build_initial_generation_prompt(problem.prompt, task_mode="function")
        best, best_reward, tokens, calls, vis = None, -1.0, 0, 0, False
        while calls < args.n and tokens < args.token_budget:
            r = await client.complete(p.system_prompt, p.user_prompt, max_tokens=args.max_tokens_per_call)
            calls, tokens = calls + 1, tokens + r.total_tokens
            rep = await fexec.run(r.text, cases)
            rew = compute_reward(rep)
            if rew > best_reward:
                best, best_reward = r.text, rew
            if rep.all_passed:
                vis = True
                break
        code = best

    else:
        if method.startswith("mcts_budget_l"):
            k, lam = args.k, float(method[len("mcts_budget_l"):])
        else:
            k, lam = {"sequential_repair": (1, 0.0), "mcts_ucb1": (args.k, 0.0), "mcts_budget": (args.k, args.lam)}[method]
        client = client_factory(args.temperature)
        search = MCTSSearch(
            llm_client=client, executor=fexec, problem_statement=problem.prompt, test_cases=cases,
            exploration_constant=args.c, token_penalty_weight=lam, total_token_budget=args.token_budget,
            max_iterations=args.n, max_tree_depth=args.max_depth, branching_factor=k, task_mode="function",
        )
        node = await search.run()
        code, tokens, calls, vis = node.state_code, search.tokens_spent, client.calls, node.is_terminal

    passed = await full_test_pass(executor, problem, code)
    return {"task_id": problem.task_id, "method": method, "passed_full": passed, "passed_visible": bool(vis),
            "llm_calls": calls, "tokens": tokens, "code": code}


# ------------------------------------------------------------ reporting
def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def summarize(rows: list[dict], out_dir: Path, run: str) -> None:
    summary = []
    for m in METHODS:
        rs = [r for r in rows if r["method"] == m]
        if not rs:
            continue
        n = len(rs)
        p = sum(r["passed_full"] for r in rs) / n
        lo, hi = wilson(p, n)
        solved = [r for r in rs if r["passed_full"]]
        overfit = sum(r["passed_visible"] and not r["passed_full"] for r in rs)
        summary.append({
            "method": m, "n": n, "pass_at_1": p, "ci_lo": lo, "ci_hi": hi,
            "mean_calls": sum(r["llm_calls"] for r in rs) / n,
            "mean_tokens": sum(r["tokens"] for r in rs) / n,
            "tokens_per_solved": (sum(r["tokens"] for r in rs) / len(solved)) if solved else float("nan"),
            "visible_pass_hidden_fail": overfit,
        })
    with open(out_dir / f"{run}_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(out_dir / f"{run}_table.tex", "w") as f:
        for s in summary:
            f.write(f"{LABELS[s['method']]} & {100*s['pass_at_1']:.1f} [{100*s['ci_lo']:.1f}, {100*s['ci_hi']:.1f}] & "
                    f"{s['mean_calls']:.2f} & {s['mean_tokens']/1000:.1f} & {s['tokens_per_solved']/1000:.1f} \\\\\n")
    for s in summary:
        print(f"{s['method']:18s} pass@1={100*s['pass_at_1']:5.1f}%  [{100*s['ci_lo']:.1f},{100*s['ci_hi']:.1f}]  "
              f"calls={s['mean_calls']:.2f}  tokens={s['mean_tokens']:.0f}  tok/solved={s['tokens_per_solved']:.0f}  "
              f"overfit={s['visible_pass_hidden_fail']}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/HumanEval.jsonl.gz")
    ap.add_argument("--backend", choices=["ollama", "openai", "fake"], default="ollama")
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--limit", type=int, default=None, help="first N problems only")
    ap.add_argument("--n", type=int, default=16, help="max model calls per problem (all methods)")
    ap.add_argument("--token-budget", type=int, default=10000, help="max tokens per problem (all methods); the binding constraint")
    ap.add_argument("--max-tokens-per-call", type=int, default=1024)
    ap.add_argument("--k", type=int, default=2, help="branching factor for MCTS")
    ap.add_argument("--lam", type=float, default=0.3, help="budget-penalty weight lambda")
    ap.add_argument("--c", type=float, default=1.41421356, help="UCB exploration constant")
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--extra-lams", default="", help="comma list of extra lambda values to run, e.g. 1.0,3.0")
    ap.add_argument("--run", default=None, help="run name (default: model_n)")
    args = ap.parse_args()

    base_url = args.base_url or ("http://localhost:11434" if args.backend == "ollama" else "http://localhost:8000/v1")
    run = args.run or f"{args.model.replace(':', '-').replace('/', '-')}_n{args.n}"
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    log = out_dir / f"{run}.jsonl"
    done = set()
    rows = []
    if log.exists():
        for line in log.open():
            r = json.loads(line)
            rows.append(r)
            done.add((r["task_id"], r["method"]))

    problems = load_humaneval(Path(args.data))[: args.limit]
    executor = CodeExecutor(SandboxLimits(timeout_seconds=6.0, memory_limit_mb=512, cpu_time_limit_seconds=8.0))
    methods = [m for m in args.methods.split(",") if m]
    for lam in [x for x in args.extra_lams.split(",") if x]:
        name = f"mcts_budget_l{lam}"
        METHODS.append(name)
        LABELS[name] = f"MCTS, budget-aware UCB1 ($\\lambda{{=}}{lam}$)"
        methods.append(name)

    for i, prob in enumerate(problems):
        for m in methods:
            if (prob.task_id, m) in done:
                continue
            if args.backend == "fake":
                factory = lambda t, s=prob.canonical: CanonicalFakeClient(s)  # noqa: E731
            else:
                factory = lambda t: HTTPClient(args.backend, args.model, base_url, t)  # noqa: E731
            try:
                r = await run_method(m, prob, factory, executor, args)
            except Exception as e:  # a crashed episode counts as a failure, not a silent skip
                r = {"task_id": prob.task_id, "method": m, "passed_full": False, "passed_visible": False,
                     "llm_calls": 0, "tokens": 0, "code": "", "error": repr(e)}
            rows.append(r)
            with log.open("a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"[{i+1}/{len(problems)}] {prob.task_id:14s} {m:18s} full={r['passed_full']} calls={r['llm_calls']}", flush=True)

    summarize([r for r in rows if r["task_id"] in {p.task_id for p in problems}], out_dir, run)


if __name__ == "__main__":
    asyncio.run(main())
