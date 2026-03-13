import sys


def _print_required(path, total_dist, total_energy):
    # Keep formatting exactly as required by the assignment.
    print("Shortest path: {}.".format("->".join(path)))
    print("Shortest distance: {}.".format(total_dist))
    print("Total energy cost: {}.".format(total_energy))


def main():
    # Make `python main.py` friendly even if `python` is Python 2 on some machines.
    if sys.version_info[0] < 3:
        print("This assignment requires Python 3. Please run: python3 main.py")
        return

    from part1.task1_dijkstra import (
        load_graph,
        load_distances,
        load_costs,
        dijkstra_shortest_path,
        compute_total_energy,
    )
    from part1.task2_search import ucs_energy_constrained
    from part1.task3_astar import astar_energy_constrained

    source = "1"
    target = "50"
    energy_budget = 287932.0  # Task 2 & 3 budget from the assignment

    # Load shared data once (big files)
    G = load_graph()
    Dist = load_distances()
    Cost = load_costs()

    # Task 1: relaxed shortest path (no energy constraint)
    d1, p1 = dijkstra_shortest_path(G, Dist, source, target)
    e1 = compute_total_energy(p1, Cost)
    _print_required(p1, d1, e1)
    print()

    # Task 2: uninformed search (UCS) with energy constraint
    d2, e2, p2 = ucs_energy_constrained(source, target, energy_budget, G, Dist, Cost)
    _print_required(p2, d2, e2)
    print()

    # Task 3: A* with heuristic + energy constraint
    d3, e3, p3 = astar_energy_constrained(source, target, energy_budget, G, None, Dist, Cost)
    _print_required(p3, d3, e3)


if __name__ == "__main__":
    main()

