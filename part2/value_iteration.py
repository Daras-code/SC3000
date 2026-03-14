from typing import Dict, Tuple

# Support both package-style and direct execution
try:  # pragma: no cover - import convenience
    from .gridworld import GridWorld, all_states, print_policy
except ImportError:  # when run as a plain script
    from gridworld import GridWorld, all_states, print_policy

State = Tuple[int, int]
Action = int


def value_iteration(gamma: float = 0.9, theta: float = 1e-4) -> Tuple[Dict[State, float], Dict[State, Action]]:
    """
    Task 1: Value Iteration on the known GridWorld model.

    Returns:
        V*: optimal state-value function
        pi*: greedy policy w.r.t V*
    """
    V: Dict[State, float] = {s: 0.0 for s in all_states()}

    while True:
        delta = 0.0
        for s in all_states():
            if GridWorld.is_terminal(s):
                continue
            v = V[s]
            best = float("-inf")
            for a in GridWorld.ACTIONS:
                q_sa = 0.0
                for s_next, p in GridWorld.get_transition_probs(s, a).items():
                    r = GridWorld.reward(s, s_next)
                    q_sa += p * (r + gamma * V[s_next])
                if q_sa > best:
                    best = q_sa
            V[s] = best
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    pi: Dict[State, Action] = {}
    for s in all_states():
        if GridWorld.is_terminal(s):
            continue
        best_a = 0
        best = float("-inf")
        for a in GridWorld.ACTIONS:
            q_sa = 0.0
            for s_next, p in GridWorld.get_transition_probs(s, a).items():
                r = GridWorld.reward(s, s_next)
                q_sa += p * (r + gamma * V[s_next])
            if q_sa > best:
                best = q_sa
                best_a = a
        pi[s] = best_a
    return V, pi


if __name__ == "__main__":
    V_opt, pi_opt = value_iteration()
    print("Optimal values (sample):")
    for y in reversed(range(GridWorld.HEIGHT)):
        row = []
        for x in range(GridWorld.WIDTH):
            s = (x, y)
            if s in GridWorld.OBSTACLES:
                row.append("  ####  ")
            else:
                row.append("{:7.2f}".format(V_opt.get(s, 0.0)))
        print(" ".join(row))
    print("\nGreedy policy from value iteration:")
    print_policy(pi_opt)

