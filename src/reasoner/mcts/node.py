"""
MCTS node for tree-search-guided code generation.

Each node represents a code draft (a full program state, not a diff) at
some point in the search. Each edge (parent -> child) represents an LLM
"action": generating a helper function, patching a specific bug, running
an optimization pass, or synthesizing an additional test case to
tighten the reward signal.

UCB1 modified for token budget
-------------------------------
Standard UCB1 balances exploitation (Q, the average backpropagated
reward) against exploration (visit-count-based uncertainty bonus):

    UCB1(s, a) = Q(s, a) + c * sqrt( ln N(s) / N(s, a) )

Every action here costs LLM tokens, and the search runs under a finite
token budget B for the whole episode (not just a move-count limit, since
token cost varies wildly between actions -- a one-line fix costs far
less than "rewrite this function from scratch"). An action that is only
marginally better in expectation but consumes a large fraction of the
remaining budget should be discounted relative to a cheap action of
similar value, or the search will blow its budget on a handful of
expensive, uncertain branches instead of exploring broadly. This adds a
budget-pressure penalty term:

    UCB1_LLM(s, a) = Q(s, a)
                    + c * sqrt( ln N(s) / N(s, a) )
                    - lambda * ( tau(s, a) / max(B_remaining, epsilon) )

Where:
    Q(s, a)       average backpropagated reward of the child reached by (s, a)
    N(s)          visit count of the parent
    N(s, a)       visit count of the child (visits to the edge)
    c             standard exploration constant (e.g. sqrt(2))
    tau(s, a)     average LLM token cost incurred generating this child
    B_remaining   tokens left in the episode's budget
    lambda        penalty weight; lambda=0 recovers standard UCB1

As B_remaining shrinks toward the end of a search episode, the penalty
term grows for any fixed tau, which is the intended effect: the search
gets more conservative about spending its dwindling budget on expensive
exploratory actions and biases toward cheap refinements of already-good
drafts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class ActionType(str, Enum):
    GENERATE_INITIAL = "generate_initial"
    FIX_BUG = "fix_bug"
    ADD_HELPER_FUNCTION = "add_helper_function"
    OPTIMIZE = "optimize"
    SYNTHESIZE_TEST_CASE = "synthesize_test_case"


@dataclass(frozen=True)
class Action:
    action_type: ActionType
    description: str
    estimated_token_cost: int = 0


@dataclass
class MCTSNode:
    state_code: str
    parent: "MCTSNode | None" = None
    incoming_action: Action | None = None
    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    untried_actions: list[Action] = field(default_factory=list)

    visit_count: int = 0
    total_value: float = 0.0
    token_cost_to_reach: int = 0

    is_terminal: bool = False
    last_failure_reason: str | None = None

    depth: int = 0

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def ucb_score(
        self,
        exploration_constant: float,
        token_budget_remaining: int,
        token_penalty_weight: float,
    ) -> float:
        if self.visit_count == 0 or self.parent is None or self.parent.visit_count == 0:
            return float("inf")

        exploitation = self.q_value
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )
        budget_pressure = token_penalty_weight * (
            self.token_cost_to_reach / max(token_budget_remaining, 1)
        )
        return exploitation + exploration - budget_pressure

    def best_child(
        self,
        exploration_constant: float,
        token_budget_remaining: int,
        token_penalty_weight: float,
    ) -> "MCTSNode":
        if not self.children:
            raise ValueError("best_child() called on a node with no children")
        return max(
            self.children.values(),
            key=lambda child: child.ucb_score(
                exploration_constant, token_budget_remaining, token_penalty_weight
            ),
        )

    def expand(self, action: Action, new_state_code: str, token_cost: int) -> "MCTSNode":
        child = MCTSNode(
            state_code=new_state_code,
            parent=self,
            incoming_action=action,
            token_cost_to_reach=token_cost,
            depth=self.depth + 1,
        )
        self.children[id(action)] = child
        if action in self.untried_actions:
            self.untried_actions.remove(action)
        return child

    def backpropagate(self, reward: float) -> None:
        node: MCTSNode | None = self
        while node is not None:
            node.visit_count += 1
            node.total_value += reward
            node = node.parent

    def path_from_root(self) -> list["MCTSNode"]:
        path = [self]
        node = self.parent
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def total_tokens_in_path(self) -> int:
        return sum(n.token_cost_to_reach for n in self.path_from_root())
