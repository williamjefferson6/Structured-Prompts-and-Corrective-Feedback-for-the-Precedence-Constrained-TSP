"""Shared helpers for the precedence-TSP project. Import this everywhere."""
import json
import hashlib
import numpy as np

DISTANCE_SCALE = 1_000_000  # integer scaling for CP-SAT distances


def euclidean_distance(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def tour_length(coords, tour):
    """Length of a closed tour with unrounded Euclidean distances."""
    coords = np.asarray(coords, dtype=float)
    return sum(euclidean_distance(coords[tour[i]], coords[tour[i + 1]])
               for i in range(len(tour) - 1))


def generate_instance(n, seed, n_precedences=None):
    """Random Euclidean instance with precedence pairs that agree with a hidden order."""
    rng = np.random.default_rng(seed)
    coords = rng.random((n, 2))
    perm = [int(c) for c in rng.permutation(np.arange(1, n))]
    hidden_order = [0] + perm + [0]
    rank = {c: i for i, c in enumerate(perm)}

    if n_precedences is None:
        n_precedences = n // 2
    max_pairs = (n - 1) * (n - 2) // 2
    n_precedences = min(n_precedences, max_pairs)

    pairs = set()
    while len(pairs) < n_precedences:
        a, b = (int(v) for v in rng.choice(np.arange(1, n), size=2, replace=False))
        if rank[a] > rank[b]:
            a, b = b, a
        pairs.add((a, b))
    precedences = sorted([list(p) for p in pairs])

    return {
        "n_cities": n,
        "seed": int(seed),
        "coordinates": coords.tolist(),
        "precedences": precedences,
        "n_precedences": len(precedences),
        "hidden_order": hidden_order,
    }


def check_tour(tour, n, precedences):
    """Strict diagnostic check. Returns {'valid': bool, 'errors': [...]}.

    Error types: not_a_list, non_integer, wrong_length, wrong_start, wrong_end,
    out_of_range, missing, duplicate, precedence.
    'missing' = a city in 1..n-1 that appears nowhere in the tour.
    'duplicate' = a city in 1..n-1 appearing more than once, or the depot 0
    appearing anywhere except first and last position.
    """
    errors = []
    if not isinstance(tour, list):
        return {"valid": False, "errors": [{"type": "not_a_list"}]}
    if not all(isinstance(c, int) and not isinstance(c, bool) for c in tour):
        return {"valid": False, "errors": [{"type": "non_integer"}]}

    if len(tour) != n + 1:
        errors.append({"type": "wrong_length", "expected": n + 1, "got": len(tour)})
    if len(tour) == 0 or tour[0] != 0:
        errors.append({"type": "wrong_start", "got": tour[0] if tour else None})
    if len(tour) == 0 or tour[-1] != 0:
        errors.append({"type": "wrong_end", "got": tour[-1] if tour else None})

    out_of_range = sorted({c for c in tour if c < 0 or c >= n})
    if out_of_range:
        errors.append({"type": "out_of_range", "cities": out_of_range})

    counts = {}
    for c in tour:
        if c != 0:
            counts[c] = counts.get(c, 0) + 1
    missing = [c for c in range(1, n) if c not in counts]
    duplicates = sorted(c for c, k in counts.items() if k > 1 and 0 <= c < n)
    if 0 in tour[1:-1]:
        duplicates = sorted(set(duplicates) | {0})
    if missing:
        errors.append({"type": "missing", "cities": missing})
    if duplicates:
        errors.append({"type": "duplicate", "cities": duplicates})

    first_pos = {}
    for i, c in enumerate(tour):
        if c != 0:
            first_pos.setdefault(c, i)
    broken = []
    for a, b in precedences:
        if a in first_pos and b in first_pos and first_pos[a] > first_pos[b]:
            broken.append([a, b])
    if broken:
        errors.append({"type": "precedence", "pairs": broken})

    return {"valid": len(errors) == 0, "errors": errors}


def coordinate_fingerprint(coords):
    payload = json.dumps(coords, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
