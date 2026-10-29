"""CP-SAT model, solver, validator and baselines for precedence TSP."""
import time
import numpy as np
from ortools.sat.python import cp_model
from tsp_utils import DISTANCE_SCALE, check_tour, tour_length


def build_model(inst, fixed_tour=None, scale=DISTANCE_SCALE):
    n = inst["n_cities"]
    C = np.asarray(inst["coordinates"])
    D = [[int(round(np.linalg.norm(C[i] - C[j]) * scale)) for j in range(n)] for i in range(n)]
    m = cp_model.CpModel()
    x = {(i, j): m.NewBoolVar(f"x_{i}_{j}") for i in range(n) for j in range(n) if i != j}
    m.AddCircuit([(i, j, v) for (i, j), v in x.items()])
    pos = [m.NewIntVar(0, n - 1, f"pos_{i}") for i in range(n)]
    m.Add(pos[0] == 0)
    for (i, j), v in x.items():
        if j != 0:
            m.Add(pos[j] == pos[i] + 1).OnlyEnforceIf(v)
    for a, b in inst["precedences"]:
        m.Add(pos[a] < pos[b])
    if fixed_tour is not None:
        for k in range(len(fixed_tour) - 1):
            m.Add(x[fixed_tour[k], fixed_tour[k + 1]] == 1)
    else:
        m.Minimize(sum(D[i][j] * v for (i, j), v in x.items()))
    return m, x


def extract_tour(solver, x, n):
    succ = {i: j for (i, j), v in x.items() if solver.Value(v)}
    tour, cur = [0], 0
    for _ in range(n):
        cur = succ[cur]
        tour.append(cur)
    return tour


def solve_cpsat(inst, time_limit=60, workers=8):
    m, x = build_model(inst)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_search_workers = workers
    s.parameters.random_seed = 0
    t0 = time.perf_counter()
    status = s.Solve(m)
    wall = time.perf_counter() - t0
    ok = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    out = {"status": s.StatusName(status), "wall_seconds": wall,
           "objective_scaled": s.ObjectiveValue() if ok else None,
           "best_bound_scaled": s.BestObjectiveBound() if ok else None}
    if ok:
        out["tour"] = extract_tour(s, x, inst["n_cities"])
        out["length"] = tour_length(inst["coordinates"], out["tour"])
        out["bound_length"] = out["best_bound_scaled"] / DISTANCE_SCALE
    return out


def validate_with_cpsat(inst, tour, time_limit=10):
    """Structural pre-check (so x[...] lookups cannot fail), then CP-SAT feasibility."""
    n = inst["n_cities"]
    if (not isinstance(tour, list) or len(tour) != n + 1
            or not all(isinstance(c, int) and not isinstance(c, bool) and 0 <= c < n for c in tour)
            or tour[0] != 0 or tour[-1] != 0
            or any(tour[k] == tour[k + 1] for k in range(len(tour) - 1))):
        return False
    m, _ = build_model(inst, fixed_tour=tour)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    status = s.Solve(m)
    return status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def nn_precedence(inst):
    """Nearest unvisited city whose prerequisites are all visited."""
    n = inst["n_cities"]
    C = np.asarray(inst["coordinates"])
    prereq = {c: set() for c in range(n)}
    for a, b in inst["precedences"]:
        prereq[b].add(a)
    visited, tour, cur = {0}, [0], 0
    while len(visited) < n:
        cands = [c for c in range(1, n) if c not in visited and prereq[c] <= visited]
        nxt = min(cands, key=lambda c: np.linalg.norm(C[cur] - C[c]))
        tour.append(nxt); visited.add(nxt); cur = nxt
    tour.append(0)
    return tour
