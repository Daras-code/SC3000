import sys


def _print_required(path, total_dist, total_energy):
    # Keep formatting exactly as required by the assignment for Part 1.
    print("Shortest path: {}.".format("->".join(path)))
    print("Shortest distance: {}.".format(total_dist))
    print("Total energy cost: {}.".format(total_energy))


def _print_value_table(V, GridWorld):
    for y in reversed(range(GridWorld.HEIGHT)):
        row = []
        for x in range(GridWorld.WIDTH):
            s = (x, y)
            if s in GridWorld.OBSTACLES:
                row.append(" ##### ")
            else:
                row.append("{:7.2f}".format(V.get(s, 0.0)))
        print(" ".join(row))


def main():
    # Make `python main.py` friendly even if `python` is Python 2 on some machines.
    if sys.version_info[0] < 3:
        print("This assignment requires Python 3.")
        print("Please run: python3 main.py")
        return

    # -------------------------
    # Part 1 imports
    # -------------------------
    from part1.task1_dijkstra import (
        load_graph,
        load_distances,
        load_costs,
        dijkstra_shortest_path,
        compute_total_energy,
    )
    from part1.task2_search import ucs_energy_constrained
    from part1.task3_astar import astar_energy_constrained

    # -------------------------
    # Part 2 imports
    # -------------------------
    
    from part2.gridworld import GridWorld, print_policy
    from part2.value_iteration import value_iteration
    from part2.policy_iteration import policy_iteration
    from part2.monte_carlo import monte_carlo_control
    from part2.q_learn import q_learning, greedy_policy_from_q

    # =========================================================
    # Part 1
    # =========================================================
    print("=" * 60)
    print("PART 1")
    print("=" * 60)

    source = "1"
    target = "50"
    energy_budget = 287932.0

    # Load shared data once
    G = load_graph()
    Dist = load_distances()
    Cost = load_costs()

    # Task 1
    print("\n[Part 1 - Task 1] Dijkstra")
    d1, p1 = dijkstra_shortest_path(G, Dist, source, target)
    e1 = compute_total_energy(p1, Cost)
    _print_required(p1, d1, e1)

    # Task 2
    print("\n[Part 1 - Task 2] UCS with energy constraint")
    d2, e2, p2 = ucs_energy_constrained(source, target, energy_budget, G, Dist, Cost)
    _print_required(p2, d2, e2)

    # Task 3
    print("\n[Part 1 - Task 3] A* with energy constraint")
    d3, e3, p3 = astar_energy_constrained(source, target, energy_budget, G, None, Dist, Cost)
    _print_required(p3, d3, e3)

    # =========================================================
    # Part 2
    # =========================================================
    print("\n" + "=" * 60)
    print("PART 2")
    print("=" * 60)

    # Task 1: Value Iteration
    print("\n[Part 2 - Task 1A] Value Iteration")
    V_vi, pi_vi = value_iteration()
    print("Optimal Value Function:")
    _print_value_table(V_vi, GridWorld)
    print("\nPolicy:")
    print_policy(pi_vi)

    # Task 1: Policy Iteration
    print("\n[Part 2 - Task 1B] Policy Iteration")
    V_pi, pi_pi = policy_iteration()
    print("Optimal Value Function:")
    _print_value_table(V_pi, GridWorld)
    print("\nPolicy:")
    print_policy(pi_pi)

    # Task 2: Monte Carlo Control
    print("\n[Part 2 - Task 2] Monte Carlo Control")
    Q_mc, pi_mc = monte_carlo_control(num_episodes=10000, epsilon=0.1, gamma=0.9)
    print("Learned Policy:")
    print_policy(pi_mc)

    # Task 3: Q-Learning
    print("\n[Part 2 - Task 3] Q-Learning")
    Q_q = q_learning(num_episodes=10000, alpha=0.1, gamma=0.9, epsilon=0.1)
    pi_q = greedy_policy_from_q(Q_q)
    print("Learned Policy:")
    print_policy(pi_q)



if __name__ == "__main__":
    main()