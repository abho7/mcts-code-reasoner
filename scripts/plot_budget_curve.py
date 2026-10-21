#!/usr/bin/env python3
"""
Fraction of HumanEval problems solved within b tokens, per method,
from a results/<run>.jsonl log written by eval_humaneval.py.

A problem counts as solved within b tokens if the method stopped because
its program passed the visible tests, spent at most b tokens getting there,
and that program also passes the full hidden suite. Because every method
stops at its first visible pass, this curve is what a run capped at b
tokens would have reported for problems solved that early.

Usage:
  python scripts/plot_budget_curve.py results/qwen2.5-coder-7b_n16.jsonl --out ../paper/budget_curve.pdf
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["best_of_n", "sequential_repair", "mcts_ucb1", "mcts_budget"]
LABEL = {
    "best_of_n": "Best-of-N",
    "sequential_repair": "Sequential repair",
    "mcts_ucb1": "MCTS-UCB1",
    "mcts_budget": "MCTS-Budget (ours)",
}
STYLE = {"best_of_n": ("o", "--"), "sequential_repair": ("s", ":"), "mcts_ucb1": ("^", "-."), "mcts_budget": ("D", "-")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default="budget_curve.pdf")
    ap.add_argument("--max-tokens", type=int, default=10000)
    ap.add_argument("--step", type=int, default=500)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.log)]
    greedy = [r for r in rows if r["method"] == "greedy"]

    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    budgets = list(range(args.step, args.max_tokens + 1, args.step))
    for m in ORDER:
        rs = [r for r in rows if r["method"] == m]
        if not rs:
            continue
        ys = [100 * sum(r["passed_visible"] and r["passed_full"] and r["tokens"] <= b for r in rs) / len(rs) for b in budgets]
        mk, ls = STYLE[m]
        ax.plot([b / 1000 for b in budgets], ys, marker=mk, linestyle=ls, markersize=2.5, linewidth=1, label=LABEL[m], color="black")
    if greedy:
        g = 100 * sum(r["passed_full"] for r in greedy) / len(greedy)
        ax.axhline(g, color="gray", linewidth=0.8, label="Greedy")
    ax.set_xlabel("Token budget per problem (thousands)")
    ax.set_ylabel("Problems solved (%)")
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
