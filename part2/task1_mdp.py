"""
Part 2 – Task 1 (MDP planning):
- Value Iteration
- Policy Iteration

Run directly with:
    python task1_mdp.py
from inside the `part2` folder.
"""

# Support both package-style and direct execution
try:  # pragma: no cover - import convenience
    from .value_iteration import value_iteration
    from .policy_iteration import policy_iteration
    from .gridworld import print_policy, GridWorld
except ImportError:  # when run as a plain script
    from value_iteration import value_iteration
    from policy_iteration import policy_iteration
    from gridworld import print_policy, GridWorld


def main():
    gamma = 0.9

    # Value Iteration
    V_vi, pi_vi = value_iteration(gamma=gamma)
    print("=== Task 1: Value Iteration ===")
    print("Optimal state-value function (V*):")
    for y in reversed(range(GridWorld.HEIGHT)):
        row = []
        for x in range(GridWorld.WIDTH):
            s = (x, y)
            if s in GridWorld.OBSTACLES:
                row.append("  ####  ")
            else:
                row.append("{:7.2f}".format(V_vi.get(s, 0.0)))
        print(" ".join(row))
    print()
    print("Optimal policy from value iteration:")
    print_policy(pi_vi)
    print()

    # Policy Iteration
    V_pi, pi_pi = policy_iteration(gamma=gamma)
    print("=== Task 1: Policy Iteration ===")
    print("Optimal state-value function (V^pi):")
    for y in reversed(range(GridWorld.HEIGHT)):
        row = []
        for x in range(GridWorld.WIDTH):
            s = (x, y)
            if s in GridWorld.OBSTACLES:
                row.append("  ####  ")
            else:
                row.append("{:7.2f}".format(V_pi.get(s, 0.0)))
        print(" ".join(row))
    print()
    print("Optimal policy from policy iteration:")
    print_policy(pi_pi)
    print()


if __name__ == "__main__":
    main()

