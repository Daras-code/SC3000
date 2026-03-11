import json
import math
import heapq
import os
from typing import Dict, List, Tuple, Set


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))


def _load_json(name: str):
    path = os.path.join(DATA_DIR, name)
    with open(path, "r") as f:
        return json.load(f)


def _load_graph() -> Dict[str, List[str]]:
    """Adjacency list: node_id -> list of neighbour node_ids (all strings)."""
    return _load_json("G.json")


def _load_coords() -> Dict[str, Tuple[float, float]]:
    """Coordinates: node_id -> (x, y)."""
    raw = _load_json("Coord.json")
    return {k: (float(v[0]), float(v[1])) for k, v in raw.items()}


def _heuristic(
    node: str, goal: str, coords: Dict[str, Tuple[float, float]]
) -> float:
    """Straight‑line (Euclidean) distance between two nodes."""
    (x1, y1) = coords[node]
    (x2, y2) = coords[goal]
    return math.hypot(x1 - x2, y1 - y2)


def greedy_best_first_search(
    start: str,
    goal: str,
    graph: Dict[str, List[str]] | None = None,
    coords: Dict[str, Tuple[float, float]] | None = None,
) -> List[str]:
    """
    Greedy best‑first search on the road network.

    Returns the sequence of node ids from start to goal (inclusive).
    Raises ValueError if no path is found.
    """
    if graph is None:
        graph = _load_graph()
    if coords is None:
        coords = _load_coords()

    if start not in graph or goal not in graph:
        raise ValueError("start or goal not in graph")

    # priority queue ordered only by heuristic value h(n)
    frontier: List[Tuple[float, str, List[str]]] = []
    start_h = _heuristic(start, goal, coords)
    heapq.heappush(frontier, (start_h, start, [start]))

    visited: Set[str] = set()

    while frontier:
        _, current, path = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return path

        for neighbour in graph.get(current, []):
            if neighbour in visited:
                continue
            h = _heuristic(neighbour, goal, coords)
            heapq.heappush(frontier, (h, neighbour, path + [neighbour]))

    raise ValueError(f"No path found from {start} to {goal}")


if __name__ == "__main__":
    # Simple manual test / example usage
    g = _load_graph()
    coords = _load_coords()
    # Example: pick two arbitrary nodes the user can change
    s, t = "1", "50"
    path = greedy_best_first_search(s, t, g, coords)
    print(f"Greedy path from {s} to {t}:")
    print(" -> ".join(path))

