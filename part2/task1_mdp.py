try:
    from .gridworld import GridWorld, print_policy
    from .value_iteration import value_iteration
    from .policy_iteration import policy_iteration
except ImportError:
    from gridworld import GridWorld, print_policy
    from value_iteration import value_iteration
    from policy_iteration import policy_iteration


def print_value_table(V):
    for y in reversed(range(GridWorld.HEIGHT)):
        row = []
        for x in range(GridWorld.WIDTH):
            s = (x, y)
            if s in GridWorld.OBSTACLES:
                row.append(" ##### ")
            else:
                row.append("{:7.2f}".format(V.get(s, 0.0)))
        print(" ".join(row))


if __name__ == "__main__":
    print("=== Value Iteration ===")
    V_vi, pi_vi = value_iteration()
    print("Value Function:")
    print_value_table(V_vi)
    print("\nPolicy:")
    print_policy(pi_vi)

    print("\n=== Policy Iteration ===")
    V_pi, pi_pi = policy_iteration()
    print("Value Function:")
    print_value_table(V_pi)
    print("\nPolicy:")
    print_policy(pi_pi)