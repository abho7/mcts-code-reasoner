from reasoner.mcts.node import Action, ActionType, MCTSNode


def test_unvisited_children_have_infinite_ucb_score():
    root = MCTSNode(state_code="")
    action = Action(ActionType.GENERATE_INITIAL, "attempt")
    child = root.expand(action, "code", token_cost=100)
    assert child.ucb_score(1.41, 100000, 0.1) == float("inf")


def test_cheap_high_reward_child_beats_expensive_low_reward_child():
    root = MCTSNode(state_code="")
    a1 = Action(ActionType.GENERATE_INITIAL, "cheap")
    a2 = Action(ActionType.GENERATE_INITIAL, "expensive")
    c1 = root.expand(a1, "code1", token_cost=500)
    c2 = root.expand(a2, "code2", token_cost=2000)
    c1.backpropagate(1.0)
    c2.backpropagate(0.3)

    s1 = c1.ucb_score(1.41, token_budget_remaining=10000, token_penalty_weight=0.5)
    s2 = c2.ucb_score(1.41, token_budget_remaining=10000, token_penalty_weight=0.5)
    assert s1 > s2
    assert root.best_child(1.41, 10000, 0.5) is c1


def test_budget_pressure_grows_as_remaining_budget_shrinks():
    root = MCTSNode(state_code="")
    action = Action(ActionType.GENERATE_INITIAL, "expensive")
    child = root.expand(action, "code", token_cost=2000)
    child.backpropagate(0.5)

    score_big_budget = child.ucb_score(1.41, token_budget_remaining=100000, token_penalty_weight=0.5)
    score_small_budget = child.ucb_score(1.41, token_budget_remaining=2500, token_penalty_weight=0.5)
    assert score_small_budget < score_big_budget


def test_zero_penalty_weight_recovers_standard_ucb1():
    root = MCTSNode(state_code="")
    action = Action(ActionType.GENERATE_INITIAL, "any")
    child = root.expand(action, "code", token_cost=999999)  # huge cost, should be irrelevant when weight=0
    child.backpropagate(0.5)

    score_no_budget_limit = child.ucb_score(1.41, token_budget_remaining=100, token_penalty_weight=0.0)
    # with weight=0, tiny remaining budget must not matter at all
    score_huge_budget = child.ucb_score(1.41, token_budget_remaining=10**9, token_penalty_weight=0.0)
    assert abs(score_no_budget_limit - score_huge_budget) < 1e-9


def test_backpropagation_updates_every_ancestor():
    root = MCTSNode(state_code="")
    a1 = Action(ActionType.GENERATE_INITIAL, "a1")
    c1 = root.expand(a1, "code1", token_cost=10)
    a2 = Action(ActionType.FIX_BUG, "a2")
    grandchild = c1.expand(a2, "code2", token_cost=10)

    grandchild.backpropagate(1.0)

    assert root.visit_count == 1
    assert c1.visit_count == 1
    assert grandchild.visit_count == 1
    assert root.total_value == 1.0
    assert c1.total_value == 1.0
    assert grandchild.total_value == 1.0


def test_path_from_root_and_total_tokens():
    root = MCTSNode(state_code="")
    a1 = Action(ActionType.GENERATE_INITIAL, "a1")
    c1 = root.expand(a1, "code1", token_cost=100)
    a2 = Action(ActionType.FIX_BUG, "a2")
    c2 = c1.expand(a2, "code2", token_cost=50)

    path = c2.path_from_root()
    assert path == [root, c1, c2]
    assert c2.total_tokens_in_path() == 150


def test_expand_removes_action_from_untried_list():
    root = MCTSNode(state_code="")
    action = Action(ActionType.GENERATE_INITIAL, "a1")
    root.untried_actions = [action]
    assert not root.is_fully_expanded()
    root.expand(action, "code", token_cost=10)
    assert root.is_fully_expanded()
