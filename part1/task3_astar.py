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
    return _load_json("G.json")


def load_coords() -> Dict[str, Tuple[float, float]]:
    raw = _load_json("Coord.json")
    return {k: (float(v[0]), float(v[1])) for k, v in raw.items()}


def load_distances() -> Dict[str, float]:
    raw = _load_json("Dist.json")
    return {k: float(v) for k, v in raw.items()}


def load_costs() -> Dict[str, float]:
    raw = _load_json("Cost.json")
    return {k: float(v) for k, v in raw.items()}


def _edge_val(d: Dict[str, float], v: str, w: str) -> float:
    return d[f"{v},{w}"]


def heuristic_euclidean(node: str, goal: str, coords: Dict[str, Tuple[float, float]]) -> float:
    (x1, y1) = coords[node]
    (x2, y2) = coords[goal]
    return math.hypot(x1 - x2, y1 - y2)


def _is_dominated(labels: List[Tuple[float, float]], energy: float, dist: float) -> bool:
    for e, d in labels:
        if e <= energy and d <= dist:
            return True
    return False


def _add_label(labels: List[Tuple[float, float]], energy: float, dist: float) -> None:
    kept: List[Tuple[float, float]] = []
    for e, d in labels:
        if not (energy <= e and dist <= d):
            kept.append((e, d))
    kept.append((energy, dist))
    labels.clear()
    labels.extend(kept)


def astar_energy_constrained(
    source: str,
    target: str,
    energy_budget: float,
    graph: Optional[Dict[str, List[str]]] = None,
    coords: Optional[Dict[str, Tuple[float, float]]] = None,
    dist: Optional[Dict[str, float]] = None,
    cost: Optional[Dict[str, float]] = None,
) -> Tuple[float, float, List[str]]:
    """
    Task 3: A* search for shortest distance subject to energy budget.

    State is (node, energy_spent) but we keep a Pareto set (energy, distance) per node
    to prune dominated labels.
    """
    if graph is None:
        graph = load_graph()
    if coords is None:
        coords = load_coords()
    if dist is None:
        dist = load_distances()
    if cost is None:
        cost = load_costs()

    nodes: List[str] = [source]
    energies: List[float] = [0.0]
    dists: List[float] = [0.0]
    parent: List[int] = [-1]

    labels: Dict[str, List[Tuple[float, float]]] = {source: [(0.0, 0.0)]}

    h0 = heuristic_euclidean(source, target, coords)
    pq: List[Tuple[float, float, int]] = [(h0, 0.0, 0)]  # (f, g, state_id)

    while pq:
        f, g_dist, sid = heapq.heappop(pq)
        v = nodes[sid]
        g_energy = energies[sid]

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

            h = heuristic_euclidean(w, target, coords)
            heapq.heappush(pq, (new_dist + h, new_dist, new_sid))

    raise ValueError(f"No feasible path from {source} to {target} within budget {energy_budget}")


if __name__ == "__main__":
    total_d, total_e, path = astar_energy_constrained("1", "50", 287932.0)
    print(total_d, total_e, len(path))

