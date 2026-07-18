# MCTS Code Reasoner

Abhineeth Duddela

[![tests](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/actions/workflows/tests.yml/badge.svg)](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/actions/workflows/tests.yml)

A test-time-compute reasoning engine for code generation: instead of a
single LLM call producing one attempt, this runs Monte Carlo Tree Search
over the space of possible code drafts, evaluating each one in a
sandboxed execution environment against real test cases, and feeding
execution failures (tracebacks, timeouts, wrong output) back into the
model as a self-reflection step to guide the next attempt.


See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full system design: the
dataflow diagram, the UCB1 formulation modified for LLM token budgets,
and a four-phase production roadmap.

## What's actually verified here, and what isn't

This matters enough to say plainly rather than let a passing test suite
imply more than it does:

**Fully implemented and tested (17/17 tests passing, zero external
dependencies required):**
- The sandboxed executor (`execution/sandbox.py`) — tested against a
  real infinite loop (confirmed killed by the wall-clock timeout, not
  just configured to be), a real memory allocation bomb (confirmed
  caught), a real `IndexError` (confirmed the traceback is captured
  correctly), and real concurrent execution (confirmed 10 test cases run
  in ~0.2s, not 10x a single case's time).
  **Platform note:** the memory (`RLIMIT_AS`) and CPU-time (`RLIMIT_CPU`)
  walls are POSIX-only — Python's `resource` module doesn't exist on
  Windows at all, and neither does `subprocess`'s `preexec_fn`. The
  sandbox detects this at import time and degrades gracefully: on
  Windows, only the wall-clock timeout is enforced. I verified this
  fallback path directly (by simulating the `resource` import failing)
  rather than just hoping the `try/except` was correct — both
  correctness checking and timeout-killing still work with it. For the
  full set of resource walls in production, run this inside a container
  or WSL rather than native Windows.
- The UCB1 token-budget math (`mcts/node.py`) — tested for its defining
  properties: unvisited nodes are prioritized, a cheap-and-better node
  beats an expensive-and-worse one, the budget-pressure term actually
  grows as remaining budget shrinks, and `λ = 0` exactly recovers
  standard UCB1.
- **The full self-correction loop, end to end** — a scripted "buggy first
  draft, then a fix" LLM response pair is run through the real
  MCTSSearch, real CodeExecutor, and real reward/reflection pipeline, and
  the search is confirmed to converge to the corrected solution in
  exactly 2 iterations, using the injected traceback to produce the fix.
  This is the core claim of the whole project, and it's the one piece
  most tempting to just assert works — it's tested instead.

**Written correctly, but not executable in the environment this repo was
built in (no network access to install packages):**
- `config.py` — the Pydantic schemas. Syntax-checked, but Pydantic's
  actual validation behavior (bound checking, the cross-field
  `model_validator`, etc.) hasn't been exercised, because Pydantic isn't
  installed here and couldn't be. `pip install -r requirements.txt` in an
  environment with network access resolves this.
- `orchestrator.py` and `llm/client.py`'s `OllamaClient`/`VLLMClient` —
  these need either Pydantic (orchestrator) or a live Ollama/vLLM server
  (the real clients) to run, neither of which exists in this sandbox.
  The *wiring logic* (does `solve_problem()` build the right client and
  drive the search correctly) was verified separately using a duck-typed
  stand-in for the config object, but the literal `RunConfig` Pydantic
  path and any real network call to a model server are unverified.

The architectural reason this split is clean rather than "everything is
untested": `mcts/search.py`, `mcts/node.py`, `execution/*`,
`reflection.py`, and `llm/client.py` (including `FakeLLMClient`) have no
Pydantic dependency at all by design. Only the outermost config/wiring
layer does. That's what made it possible to fully test the actual
reasoning algorithm without a working Python package index.

## Quickstart

```bash
pip install -r requirements.txt

# Smoke test with no model server required -- scripted LLM responses:
python scripts/run_example.py --fake

# Against a real local model via Ollama:
ollama pull qwen2.5-coder:32b
ollama serve
python scripts/run_example.py --backend ollama --model qwen2.5-coder:32b
```

Run the test suite:
```bash
pytest tests/ -v
```

## Project layout

See the "Production Repository Structure" section in
[ARCHITECTURE.md](./ARCHITECTURE.md) for the annotated directory tree.
