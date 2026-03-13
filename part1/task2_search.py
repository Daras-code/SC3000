import json
import heapq
import math
import os
from typing import Dict, List, Tuple, Optional


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))


def _load_json(name: str):
    path = os.path.join(DATA_DIR, name)
    with open(path, "r") as f:
        return json.load(f)


def load_graph() -> Dict[str, List[str]]:
    """Adjacency list: node_id -> list of neighbour node_ids (all strings)."""
    return _load_json("G.json")


def load_coords() -> Dict[str, Tuple[float, float]]:
    """Coordinates: node_id -> (x, y)."""
    raw = _load_json("Coord.json")
    return {k: (float(v[0]), float(v[1])) for k, v in raw.items()}


def load_distances() -> Dict[str, float]:
    """Edge distance dictionary Dist with keys like 'v,w'."""
    raw = _load_json("Dist.json")
    return {k: float(v) for k, v in raw.items()}


def load_costs() -> Dict[str, float]:
    """Edge energy cost dictionary Cost with keys like 'v,w'."""
    raw = _load_json("Cost.json")
    return {k: float(v) for k, v in raw.items()}


def _edge_val(d: Dict[str, float], v: str, w: str) -> float:
    return d[f"{v},{w}"]


def _heuristic_euclidean(
    node: str, goal: str, coords: Dict[str, Tuple[float, float]]
) -> float:
    """Straight-line (Euclidean) distance heuristic (used by greedy demo)."""
    (x1, y1) = coords[node]
    (x2, y2) = coords[goal]
    return math.hypot(x1 - x2, y1 - y2)


def greedy_best_first_search(
    source: str,
    target: str,
    graph: Optional[Dict[str, List[str]]] = None,
    coords: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[str]:
    """
    Greedy best-first search (not used for the assignment's Task 2, kept for reference).
    """
    if graph is None:
        graph = load_graph()
    if coords is None:
        coords = load_coords()

    pq: List[Tuple[float, str, List[str]]] = []
    heapq.heappush(pq, (_heuristic_euclidean(source, target, coords), source, [source]))
    visited = set()

    while pq:
        _, v, path = heapq.heappop(pq)
        if v in visited:
            continue
        visited.add(v)
        if v == target:
            return path
        for w in graph.get(v, []):
            if w in visited:
                continue
            heapq.heappush(pq, (_heuristic_euclidean(w, target, coords), w, path + [w]))

    raise ValueError(f"No path found from {source} to {target}")


def _is_dominated(labels: List[Tuple[float, float]], energy: float, dist: float) -> bool:
    """
    True if (energy, dist) is dominated by an existing label:
    exists (e, d) with e <= energy and d <= dist.
    """
    for e, d in labels:
        if e <= energy and d <= dist:
            return True
    return False


def _add_label(labels: List[Tuple[float, float]], energy: float, dist: float) -> None:
    """Insert (energy, dist) and remove any labels it dominates."""
    kept: List[Tuple[float, float]] = []
    for e, d in labels:
        # remove (e,d) if new label dominates it
        if not (energy <= e and dist <= d):
            kept.append((e, d))
    kept.append((energy, dist))
    labels.clear()
    labels.extend(kept)


def ucs_energy_constrained(
    source: str,
    target: str,
    energy_budget: float,
    graph: Optional[Dict[str, List[str]]] = None,
    dist: Optional[Dict[str, float]] = None,
    cost: Optional[Dict[str, float]] = None,
) -> Tuple[float, float, List[str]]:
    """
    Task 2: Uniform Cost Search (uninformed) for shortest distance subject to energy budget.

    Returns:
        (total_distance, total_energy, path)
    """
    if graph is None:
        graph = load_graph()
    if dist is None:
        dist = load_distances()
    if cost is None:
        cost = load_costs()

    # State arrays (for compact backpointers)
    nodes: List[str] = [source]
    energies: List[float] = [0.0]
    dists: List[float] = [0.0]
    parent: List[int] = [-1]

    # Non-dominated labels per node: list of (energy, dist)
    labels: Dict[str, List[Tuple[float, float]]] = {source: [(0.0, 0.0)]}

    pq: List[Tuple[float, int]] = [(0.0, 0)]  # (g_dist, state_id)

    while pq:
        g_dist, sid = heapq.heappop(pq)
        v = nodes[sid]
        g_energy = energies[sid]

        # Skip if this label got dominated after insertion
        if _is_dominated(labels.get(v, []), g_energy, g_dist) and (g_energy, g_dist) not in labels.get(v, []):
            continue

        if v == target:
            path: List[str] = []
            cur = sid
            while cur != -1:
                path.append(nodes[cur])
                cur = parent[cur]
            path.reverse()
            return g_dist, g_energy, path

        for w in graph.get(v, []):
            try:
                edge_d = _edge_val(dist, v, w)
                edge_c = _edge_val(cost, v, w)
            except KeyError:
                continue

            new_energy = g_energy + edge_c
            if new_energy > energy_budget:
                continue

            new_dist = g_dist + edge_d

            w_labels = labels.setdefault(w, [])
            if _is_dominated(w_labels, new_energy, new_dist):
                continue
            _add_label(w_labels, new_energy, new_dist)

            new_sid = len(nodes)
            nodes.append(w)
            energies.append(new_energy)
            dists.append(new_dist)
            parent.append(sid)
            heapq.heappush(pq, (new_dist, new_sid))

    raise ValueError(f"No feasible path from {source} to {target} within budget {energy_budget}")


if __name__ == "__main__":
    # Quick sanity run using assignment's Task 2 settings.
    total_d, total_e, path = ucs_energy_constrained("1", "50", 287932.0)
    print(total_d, total_e, len(path))

