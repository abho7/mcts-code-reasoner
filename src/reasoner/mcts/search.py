"""
The async MCTS driver loop: select -> expand -> simulate -> backpropagate,
repeated until a fully-passing draft is found, the token budget is spent,
or the iteration cap is hit.

Deliberately takes primitive config values (not the Pydantic RunConfig)
so this module -- the actual search algorithm -- has no hard dependency
on pydantic being installed. reasoner.orchestrator is the layer that
reads a validated RunConfig and unpacks it into this constructor; that
keeps the algorithm testable in isolation.
"""

from __future__ import annotations

import logging

from reasoner.execution.reward import compute_reward, summarize_failure
from reasoner.execution.sandbox import CodeExecutor, ExecutionReport
from reasoner.llm.client import LLMClient
from reasoner.mcts.node import Action, ActionType, MCTSNode
from reasoner.reflection import build_initial_generation_prompt, build_reflection_prompt

logger = logging.getLogger(__name__)


def _extract_code(llm_text: str) -> str:
    """Strip markdown fences if the model added them despite instructions
    not to -- cheap defensive parsing, since 'ignore the fence' is a lot
    more robust than 'hope the model never adds one'."""
    text = llm_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


class MCTSSearch:
    def __init__(
        self,
        llm_client: LLMClient,
        executor: CodeExecutor,
        problem_statement: str,
        test_cases: list[dict],
        *,
        exploration_constant: float = 1.41421356,
        token_penalty_weight: float = 0.3,
        total_token_budget: int = 200_000,
        max_iterations: int = 64,
        max_tree_depth: int = 12,
        reward_success_threshold: float = 1.0,
    ):
        self.llm_client = llm_client
        self.executor = executor
        self.problem_statement = problem_statement
        self.test_cases = test_cases
        self.exploration_constant = exploration_constant
        self.token_penalty_weight = token_penalty_weight
        self.total_token_budget = total_token_budget
        self.max_iterations = max_iterations
        self.max_tree_depth = max_tree_depth
        self.reward_success_threshold = reward_success_threshold

        self.tokens_spent = 0
        self.iterations_run = 0
        self.root = MCTSNode(state_code="")
        self.root.untried_actions = [Action(ActionType.GENERATE_INITIAL, "initial full solution attempt")]
        self.best_node: MCTSNode | None = None

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.total_token_budget - self.tokens_spent)

    def _select(self) -> MCTSNode:
        """Walk from root via UCB1_LLM until hitting a node that still has
        untried actions (i.e. is expandable) or has no children at all."""
        node = self.root
        while node.is_fully_expanded() and not node.is_leaf() and node.depth < self.max_tree_depth:
            node = node.best_child(self.exploration_constant, self.tokens_remaining, self.token_penalty_weight)
        return node

    async def _generate_action_code(self, node: MCTSNode, action: Action) -> tuple[str, int]:
        if action.action_type == ActionType.GENERATE_INITIAL:
            prompt = build_initial_generation_prompt(self.problem_statement)
        elif action.action_type == ActionType.FIX_BUG:
            prompt = build_reflection_prompt(
                self.problem_statement,
                node.state_code,
                node.last_failure_reason or "No failure detail recorded.",
                prior_attempts_summary=self._sibling_failure_notes(node),
            )
        else:
            # ADD_HELPER_FUNCTION / OPTIMIZE / SYNTHESIZE_TEST_CASE share the
            # reflection-style prompt shape for now; a real system would give
            # each its own prompt builder the way FIX_BUG and GENERATE_INITIAL do.
            prompt = build_reflection_prompt(
                self.problem_statement, node.state_code,
                f"Requested action: {action.action_type.value} -- {action.description}",
            )

        response = await self.llm_client.complete(prompt.system_prompt, prompt.user_prompt)
        return _extract_code(response.text), response.total_tokens

    def _sibling_failure_notes(self, node: MCTSNode) -> str | None:
        """What else has this branch's ancestors already tried and failed
        at? Lets reflection prompts avoid repeating a sibling branch's
        mistake instead of just re-litigating the same fix."""
        notes = [
            n.last_failure_reason for n in node.path_from_root()
            if n.last_failure_reason and n is not node
        ]
        return "\n---\n".join(notes) if notes else None

    async def _evaluate(self, code: str) -> tuple[float, ExecutionReport]:
        report = await self.executor.run(code, self.test_cases)
        return compute_reward(report), report

    async def step(self) -> MCTSNode | None:
        """Run one select/expand/simulate/backprop cycle. Returns the
        expanded node, or None if the search has nothing left to expand
        (e.g. max depth reached everywhere)."""
        selected = self._select()

        if selected.is_terminal or not selected.untried_actions:
            return None

        action = selected.untried_actions[0]
        code, token_cost = await self._generate_action_code(selected, action)
        self.tokens_spent += token_cost

        child = selected.expand(action, code, token_cost)
        reward, report = await self._evaluate(code)

        if report.all_passed:
            child.is_terminal = True
        else:
            child.last_failure_reason = summarize_failure(report)
            # A non-terminal, non-empty-code node always gets a FIX_BUG
            # option available for future expansion -- this is what keeps
            # the tree able to self-correct rather than dead-ending.
            if child.depth < self.max_tree_depth:
                child.untried_actions = [Action(ActionType.FIX_BUG, f"fix: {report.first_failure.status.value if report.first_failure else 'unknown failure'}")]

        child.backpropagate(reward)

        if self.best_node is None or child.q_value > self.best_node.q_value:
            self.best_node = child

        self.iterations_run += 1
        return child

    async def run(self) -> MCTSNode:
        """Runs step() until success, budget exhaustion, or iteration cap.
        Returns the best node found (by q_value), which is the terminal
        success node if one was found."""
        while self.iterations_run < self.max_iterations and self.tokens_remaining > 0:
            child = await self.step()
            if child is None:
                logger.info("Search exhausted expandable nodes at iteration %d", self.iterations_run)
                break
            if child.is_terminal and child.q_value >= self.reward_success_threshold:
                logger.info("Converged to a fully-passing solution at iteration %d", self.iterations_run)
                return child

        if self.best_node is None:
            raise RuntimeError("Search ran but never expanded a single node -- check test_cases/problem_statement are non-empty.")
        return self.best_node
