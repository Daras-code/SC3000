import json
import heapq
import os
from typing import Dict, List, Tuple, Optional


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))


def _load_json(name: str):
    path = os.path.join(DATA_DIR, name)
    with open(path, "r") as f:
        return json.load(f)


def load_graph() -> Dict[str, List[str]]:
    """Adjacency list of the NYC road network: node_id -> list of neighbour node_ids."""
    return _load_json("G.json")


def load_distances() -> Dict[str, float]:
    """
    Edge distance dictionary Dist where the distance between (v, w)
    is stored under key "v,w".
    """
    raw = _load_json("Dist.json")
    # Ensure values are floats (JSON may store them as strings)
    return {k: float(v) for k, v in raw.items()}


def load_costs() -> Dict[str, float]:
    """
    Edge energy cost dictionary Cost where the cost between (v, w)
    is stored under key "v,w".
    """
    raw = _load_json("Cost.json")
    return {k: float(v) for k, v in raw.items()}


def _edge_dist(
    dist: Dict[str, float],
    v: str,
    w: str,
) -> float:
    """Convenience accessor for Dist['v,w']."""
    key = f"{v},{w}"
    return dist[key]


def _edge_cost(
    cost: Dict[str, float],
    v: str,
    w: str,
) -> float:
    """Convenience accessor for Cost['v,w']."""
    key = f"{v},{w}"
    return cost[key]


def dijkstra_shortest_path(
    graph: Dict[str, List[str]],
    dist: Dict[str, float],
    source: str,
    target: str,
) -> Tuple[float, List[str]]:
    """
    Dijkstra's algorithm for the single‑source shortest path problem.

    - graph: adjacency list G
    - dist: edge distance dictionary Dist (keys "v,w")
    - source: starting node id
    - target: goal node id

    Returns:
        (total_distance, path_as_list_of_node_ids)
    Raises:
        ValueError if target is unreachable from source.
    """
    if source not in graph or target not in graph:
        raise ValueError("source or target not in graph")

    # Min‑heap of (current_distance, node_id)
    frontier: List[Tuple[float, str]] = [(0.0, source)]

    # Best known distances and predecessors
    best_dist: Dict[str, float] = {source: 0.0}
    prev: Dict[str, Optional[str]] = {source: None}

    visited = set()

    while frontier:
        current_dist, v = heapq.heappop(frontier)

        if v in visited:
            continue
        visited.add(v)

        # Early exit when we reach the target
        if v == target:
            break

        for w in graph.get(v, []):
            if w in visited:
                continue
            try:
                edge_len = _edge_dist(dist, v, w)
            except KeyError:
                # In case Dist is missing an edge that exists in G, skip it.
                continue

            new_dist = current_dist + edge_len
            if w not in best_dist or new_dist < best_dist[w]:
                best_dist[w] = new_dist
                prev[w] = v
                heapq.heappush(frontier, (new_dist, w))

    if target not in best_dist:
        raise ValueError(f"No path from {source} to {target}")

    # Reconstruct path from target back to source
    path: List[str] = []
    node: Optional[str] = target
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()

    return best_dist[target], path


def compute_total_energy(
    path: List[str],
    cost: Dict[str, float],
) -> float:
    """Sum energy cost along a given path using Cost['v,w']."""
    total = 0.0
    for i in range(len(path) - 1):
        v = path[i]
        w = path[i + 1]
        total += _edge_cost(cost, v, w)
    return total


if __name__ == "__main__":
    # Example usage / quick manual test.
    G = load_graph()
    Dist = load_distances()
    Cost = load_costs()

    # You can change these to any valid node ids from the dataset.
    s = "1"
    t = "50"

    total_d, path = dijkstra_shortest_path(G, Dist, s, t)
    total_energy = compute_total_energy(path, Cost)

    # Required output format, e.g.:
    # Shortest path: S->1->T.
    # Shortest distance: 12.
    # Total energy cost: 10.
    print(f"Shortest path: {'->'.join(path)}.")
    print(f"Shortest distance: {total_d}.")
    print(f"Total energy cost: {total_energy}.")

