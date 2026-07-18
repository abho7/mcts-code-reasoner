# Architecture: System-2 Test-Time Compute Reasoning Engine for Code Generation

## 1. System Architecture & Mathematics

### 1.1 Dataflow diagram

```
                              ┌─────────────────────────┐
                              │   Problem Statement       │
                              │   + Test Cases (JSON)      │
                              └────────────┬─────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │   MCTSSearch.run()        │
                              │   root = empty draft       │
                              │   root.untried_actions =    │
                              │     [GENERATE_INITIAL]       │
                              └────────────┬─────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     │            MCTS ITERATION LOOP              │
                     │      (repeats until success / budget / cap)  │
                     └─────────────────────┬─────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────┐
                    │  1. SELECT                                     │
                    │     Walk root → leaf via UCB1_LLM(s,a):          │
                    │       argmax  Q(s,a) + c·√(ln N(s)/N(s,a))         │
                    │                    − λ·(τ(s,a) / B_remaining)      │
                    │     Stops at first node with untried actions        │
                    │     or with no children at all.                      │
                    └──────────────────────┬──────────────────────┘
                                           │  selected node
                    ┌──────────────────────▼──────────────────────┐
                    │  2. EXPAND                                      │
                    │     Pop one untried Action (e.g. FIX_BUG).        │
                    │     Build prompt:                                  │
                    │       - GENERATE_INITIAL → problem statement only    │
                    │       - FIX_BUG → prior code + injected failure       │
                    │                    (traceback / pass-rate / timeout)   │
                    │     Call LLMClient.complete(system, user) → new draft   │
                    │     Create child MCTSNode(state_code=new draft)          │
                    └──────────────────────┬──────────────────────┘
                                           │  new code draft
                    ┌──────────────────────▼──────────────────────┐
                    │  3. SIMULATE (sandboxed evaluation)              │
                    │     CodeExecutor.run(code, test_cases)             │
                    │       → spawns one subprocess PER test case,         │
                    │         all concurrently (asyncio.gather)              │
                    │       → each subprocess walled by:                      │
                    │            RLIMIT_AS   (memory)                          │
                    │            RLIMIT_CPU  (cpu time)                         │
                    │            asyncio.wait_for (wall-clock backstop)          │
                    │     → ExecutionReport (per-test pass/fail + tracebacks)      │
                    │     compute_reward(report) → scalar in [0,1]                  │
                    └──────────────────────┬──────────────────────┘
                                           │  reward, failure detail
                    ┌──────────────────────▼──────────────────────┐
                    │  4. SELF-REFLECTION (if not fully passing)       │
                    │     summarize_failure(report) → verbal failure      │
                    │       summary (traceback tail / timeout note /        │
                    │       wrong-output diff / pass-rate)                    │
                    │     Stored on child.last_failure_reason                  │
                    │     child.untried_actions = [FIX_BUG]                     │
                    │       → this is what's injected into the NEXT              │
                    │         iteration's expand-phase prompt if this              │
                    │         branch gets selected again                            │
                    └──────────────────────┬──────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────┐
                    │  5. BACKPROPAGATE                                │
                    │     child.backpropagate(reward):                   │
                    │       walks child → parent → ... → root,             │
                    │       incrementing visit_count and total_value          │
                    │       at every ancestor.                                  │
                    │     This is what lets UCB1_LLM's Q(s,a) and N(s,a)          │
                    │       reflect the outcome on the NEXT selection phase.       │
                    └──────────────────────┬──────────────────────┘
                                           │
                              ┌────────────▼─────────────┐
                              │  reward ≥ threshold?         │
                              │  budget exhausted?             │
                              │  iteration cap hit?              │
                              └────┬─────────────────┬────────┘
                                  │ no                 │ yes
                                  │ (loop)              ▼
                                  │          ┌───────────────────────┐
                                  └─────────▶│  Return best_node        │
                                             │  (highest q_value found,   │
                                             │   terminal if converged)     │
                                             └───────────────────────┘
```

### 1.2 UCB1 modified for LLM token budget

Standard UCB1 balances exploitation against exploration using only visit
counts:

```
UCB1(s, a) = Q(s, a) + c · sqrt( ln N(s) / N(s, a) )
```

This project's search runs under a **finite token budget** for the whole
episode, and different actions cost wildly different amounts of budget
(a one-line patch vs. "rewrite this function"). An unmodified UCB1 has
no way to prefer a cheap, slightly-worse-in-expectation action over an
expensive, slightly-better one — it will happily let the search blow its
entire budget probing a handful of expensive branches. The modification
adds a budget-pressure term:

```
UCB1_LLM(s, a) = Q(s, a)
                + c · sqrt( ln N(s) / N(s, a) )
                − λ · ( τ(s, a) / max(B_remaining, ε) )
```

| Symbol | Meaning |
|---|---|
| `Q(s, a)` | average backpropagated reward of the child reached via action `a` from state `s` |
| `N(s)` | visit count of the parent node |
| `N(s, a)` | visit count of the child (visits to this edge) |
| `c` | standard exploration constant (default `√2`) |
| `τ(s, a)` | token cost incurred generating this child (from the LLM call's actual usage) |
| `B_remaining` | tokens left in the episode's total budget |
| `λ` | penalty weight; `λ = 0` recovers standard UCB1 exactly |

As `B_remaining` shrinks over the course of a search episode, the penalty
term grows for any fixed `τ`, so the search becomes progressively more
conservative about spending its dwindling budget on expensive exploratory
actions late in an episode, and biases toward cheap refinements of
already-promising drafts. This is implemented and unit-tested in
`src/reasoner/mcts/node.py::MCTSNode.ucb_score` — see
`tests/test_mcts_node.py` for the specific property tests (cheap-beats-
expensive at equal reward, budget-pressure monotonicity, `λ=0` recovers
vanilla UCB1).

### 1.3 Reward shaping

Pass/fail alone loses information: a draft passing 8/10 test cases and
one crashing on the first line are both "failures" but represent very
different progress. The reward function (`execution/reward.py`) uses
pass-rate as the base signal, minus a small structural penalty for
failure categories that indicate the draft is actively broken (crash,
timeout, OOM) rather than just logically wrong on some inputs — these
often need a different fix strategy, and the distinction is preserved in
`last_failure_reason` so the reflection prompt can be specific about it.

---

## 2. Production Repository Structure

```
mcts-code-reasoner/
├── src/reasoner/
│   ├── config.py              # Pydantic schemas: LLMConfig, SandboxConfig, MCTSConfig, RunConfig
│   ├── orchestrator.py        # top-level driver: RunConfig → concrete clients → MCTSSearch
│   ├── logging_utils.py       # structured logging setup
│   ├── reflection.py          # self-reflection / initial-generation / optimization prompt builders
│   ├── mcts/
│   │   ├── node.py             # MCTSNode, Action, UCB1_LLM scoring
│   │   └── search.py            # the async select/expand/simulate/backprop loop
│   ├── execution/
│   │   ├── sandbox.py           # CodeExecutor: subprocess isolation, RLIMIT_AS/CPU, wall-clock timeout
│   │   └── reward.py            # ExecutionReport → scalar reward + verbal failure summary
│   └── llm/
│       └── client.py            # LLMClient ABC + OllamaClient + VLLMClient + FakeLLMClient (test double)
├── tests/
│   ├── test_sandbox.py          # timeout/memory/correctness/concurrency, run against real subprocesses
│   ├── test_mcts_node.py         # UCB1 math properties
│   └── test_search_integration.py  # full self-correction loop, driven by FakeLLMClient
├── problems/
│   └── two_sum_indices.json     # example problem in the expected schema
├── scripts/
│   └── run_example.py           # CLI demo entry point (--fake smoke test or a live model backend)
├── requirements.txt
├── README.md
└── ARCHITECTURE.md              # this file
```

**Design principle behind the split:** `mcts/search.py`, `mcts/node.py`,
`execution/sandbox.py`, `execution/reward.py`, `reflection.py`, and
`llm/client.py` have **zero dependency on Pydantic** — they take
primitive types (`float`, `int`, plain dicts) in their constructors. Only
`config.py` and `orchestrator.py` know about Pydantic. This means the
actual search algorithm is unit-testable in any environment, even one
where installing Pydantic isn't possible, while the outer layer still
gets strict runtime validation for anything a human or a config file
actually sets.

---

## 3. Four-Phase Production Roadmap

### Phase 1 — Correctness (current state of this repo)
Get the algorithm right in isolation before worrying about scale:
- ✅ Sandboxed execution with real resource walls (memory/CPU/wall-clock), tested against actual timeout/OOM/crash scenarios, not just happy-path code
- ✅ UCB1_LLM selection with the token-budget penalty term, unit-tested for its defining properties
- ✅ End-to-end self-correction loop verified against a scripted LLM standing in for a real model, so the *search logic* is proven correct independent of *model quality*
- Not yet done: wiring against a real Ollama/vLLM server (needs a running model backend this environment doesn't have)

### Phase 2 — Latency: parallelize what's independent
The two genuinely slow operations are LLM calls and sandbox execution.
Within one MCTS iteration these are sequential (you need the code before
you can run it), but **across nodes** they're not:
- **Batch node evaluation**: instead of one select→expand→simulate cycle
  at a time, select the top-K nodes by UCB1_LLM each round and expand/
  evaluate them concurrently with `asyncio.gather`. `CodeExecutor.run`
  already runs all test cases for one draft concurrently — extending this
  to run multiple drafts' full evaluations concurrently is the same
  pattern one level up, bounded by `SandboxConfig.max_concurrent_executions`.
- **Prompt-prefix caching**: `GENERATE_INITIAL` and `FIX_BUG` prompts
  share a large common prefix (the problem statement, the system prompt).
  vLLM's automatic prefix caching (and Ollama's KV-cache reuse) already
  captures much of this for free if the shared prefix is placed first and
  stays byte-identical across calls — a reason the reflection prompt
  builder puts the problem statement before the variable failure detail,
  not after.
- **Speculative expansion**: while waiting on a slow sandbox run for one
  node, the LLM can already be generating the next candidate action for a
  different node — this is naturally expressible as multiple concurrent
  `step()` coroutines sharing the same tree, guarded by a lock around
  tree mutation (selection + backprop), which is the next concurrency
  primitive to add to `MCTSSearch`.

### Phase 3 — Cost: token-budget-aware search discipline
- The `token_penalty_weight` (λ) parameter exists exactly for this: raise
  it in production to make the search converge faster and cheaper at some
  cost to solution quality, or anneal it upward over the course of a
  single episode (start exploratory, end conservative) rather than
  holding it fixed.
- Cap `max_tokens_per_call` tightly for `FIX_BUG` actions specifically —
  a targeted patch rarely needs the full generation budget a from-scratch
  `GENERATE_INITIAL` does, and asymmetric budgets per action type are a
  cheap win not yet implemented here.
- Early-exit on partial credit plateaus: if `reward` hasn't improved over
  the last N expansions of a branch, that's a signal to deprioritize it
  via the UCB1 exploration term naturally, but a hard "stop expanding a
  branch that's plateaued" rule can save budget the soft UCB pressure
  alone won't catch quickly enough.

### Phase 4 — Scale: many problems, not just one
- **Cross-episode caching**: two different problems sharing a subproblem
  (e.g. "implement a segment tree") shouldn't pay the generation cost
  twice — a content-addressed cache keyed on a semantic fingerprint of
  the requested helper function is the highest-leverage addition here,
  not yet implemented.
- **Process-pool sandbox isolation instead of per-call subprocess spawn**:
  subprocess spawn overhead (~10-20ms) is negligible per-call but adds up
  at scale (thousands of problems/hour) — a warm worker-process pool that
  accepts code over a pipe and enforces the same RLIMIT walls per-task
  amortizes that cost.
- **Horizontal scaling of search episodes**: since each `MCTSSearch`
  instance owns its own tree and doesn't share state with others, running
  many episodes concurrently (one problem each) parallelizes trivially
  across processes/machines — the bottleneck becomes LLM server
  throughput, not search-algorithm design, which is why Phase 2's
  prefix-caching and batching work is the higher-leverage investment
  before scaling instance count.
