# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import pulp

def solve_mixed_assign_problem(
    o2o_cost: torch.Tensor,
    o2m_cost: torch.Tensor,
    pred_o2m_possible_idx,
    gt_subsets
):
    """
    Solve the mixed (one-to-one) + (one-to-many) assignment problem:
      - o2o_cost: [N, M] float tensor (cost of pred_i -> gt_j, 1-to-1)
      - o2m_cost: [len(pred_o2m_possible_idx), S] float tensor
                  (cost of pred -> subset)
      - pred_o2m_possible_idx: list of predictions that can do one-to-many
      - gt_subsets: list of lists with GT indices in each subset
    Returns a dict with:
      - "one2one": a list of (i, j) pairs assigned in one-to-one
      - "one2subset": a list of (i_global, s_idx) pairs for chosen subsets
      - "cost": minimal total cost
    """

    # Dimensions
    N, M = o2o_cost.shape
    P = len(pred_o2m_possible_idx)
    S = len(gt_subsets)

    # Create a PuLP problem
    # LpMinimize indicates we are minimizing the objective
    problem = pulp.LpProblem("one_to_many_assignment", pulp.LpMinimize)

    # --- 1) Create Variables ---

    # x[i][j]: binary indicating pred i assigned to gt j (one-to-one)
    x = {}
    for i in range(N):
        for j in range(M):
            x[(i, j)] = pulp.LpVariable(
                f"x_{i}_{j}", cat=pulp.LpBinary
            )

    # y[p][s]: binary for pred p (in pred_o2m_possible_idx) assigned to subset s
    # p is local index in [0..P-1], map to actual pred index via pred_o2m_possible_idx[p]
    y = {}
    for p in range(P):
        for s_idx in range(S):
            y[(p, s_idx)] = pulp.LpVariable(
                f"y_{p}_{s_idx}", cat=pulp.LpBinary
            )

    # --- 2) Build Constraints ---

    # (a) Each ground truth j must be covered exactly once
    for j in range(M):
        # sum of x[i, j] over all i
        # plus sum of y[p, s] for all subsets s containing j
        problem += (
            pulp.lpSum(x[(i, j)] for i in range(N)) +
            pulp.lpSum(y[(p, s_idx)]
                       for p in range(P)
                       for s_idx, subset in enumerate(gt_subsets) 
                       if j in subset)
        ) == 1, f"cover_gt_{j}"

    # (b) Predictions not in pred_o2m_possible_idx can match at most one ground truth
    set_o2m = set(pred_o2m_possible_idx)  # for quick membership
    for i in range(N):
        if i not in set_o2m:
            problem += (
                pulp.lpSum(x[(i, j)] for j in range(M)) <= 1
            ), f"pred_{i}_o2o_only"

    # (c) (Optional) For each multi-capable prediction, at most one subset
    # If you want to allow multiple distinct subsets (only if they do not overlap),
    # remove or modify this constraint. Here we'll keep it for clarity:
    for p in range(P):
        problem += (
            pulp.lpSum(y[(p, s_idx)] for s_idx in range(S)) <= 1
        ), f"pred_{p}_at_most_one_subset"

    # (d) Disallow double-coverage of the same GT j by the same multi-capable pred
    # i.e., x[i][j] + sum_{s_idx: j in subset[s_idx]} y[p, s_idx] <= 1
    # for i = pred_o2m_possible_idx[p]
    for p in range(P):
        i_global = pred_o2m_possible_idx[p]
        for j in range(M):
            subsets_covering_j = [s_idx for s_idx, subset in enumerate(gt_subsets) if j in subset]
            if subsets_covering_j:
                problem += (
                    x[(i_global, j)] +
                    pulp.lpSum(y[(p, s_idx)] for s_idx in subsets_covering_j) <= 1
                ), f"no_double_cover_pred_{i_global}_gt_{j}"

    # --- 3) Objective: minimize total cost ---
    # sum of o2o_cost[i,j] * x[i,j] + sum of o2m_cost[p,s] * y[p,s]
    objective_terms = []

    # one-to-one costs
    for i in range(N):
        for j in range(M):
            c = float(o2o_cost[i, j].item())
            if torch.isinf(o2o_cost[i, j]):
                # If one-to-one cost is inf, forbid x[i,j]
                problem += (x[(i, j)] == 0)
            else:
                objective_terms.append(c * x[(i, j)])

    # one-to-many costs
    for p in range(P):
        for s_idx in range(S):
            cost_val = float(o2m_cost[p, s_idx].item())
            if torch.isinf(o2m_cost[p, s_idx]):
                # If cost is inf, forbid y[p, s_idx]
                problem += (y[(p, s_idx)] == 0)
            else:
                objective_terms.append(cost_val * y[(p, s_idx)])

    problem += pulp.lpSum(objective_terms), "total_cost"

    # --- 4) Solve ---
    # You can choose a solver if you like (e.g. pulp.COIN_CMD, etc.)
    # If you don't specify, PuLP tries the default available solver.
    result_status = problem.solve(pulp.PULP_CBC_CMD(msg=0))  
    # "msg=0" to silence solver output. Remove if you want logs.

    if pulp.LpStatus[result_status] not in ("Optimal", "Feasible"):
        raise RuntimeError(f"No feasible solution found (status={pulp.LpStatus[result_status]}).")

    # --- 5) Extract solution ---
    # One-to-one assignments
    chosen_o2o = []
    for (i, j), var in x.items():
        if pulp.value(var) > 0.5:
            chosen_o2o.append((i, j))

    # One-to-many subsets
    chosen_o2m = []
    for (p, s_idx), var in y.items():
        if pulp.value(var) > 0.5:
            i_global = pred_o2m_possible_idx[p]
            chosen_o2m.append((i_global, s_idx))

    solution = {
        "one2one": chosen_o2o,
        "one2subset": chosen_o2m,
        "costs": pulp.value(problem.objective)
    }
    return solution
