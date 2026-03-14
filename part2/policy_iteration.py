from typing import Dict, Tuple

try:
    from .gridworld import GridWorld, all_states, print_policy
except ImportError:
    from gridworld import GridWorld, all_states, print_policy

State = Tuple[int, int]
Action = int


def policy_evaluation(
    pi: Dict[State, Action],
    gamma: float = 0.9,
    theta: float = 1e-4
) -> Dict[State, float]:
    V: Dict[State, float] = {s: 0.0 for s in all_states()}

    while True:
        delta = 0.0

        for s in all_states():
            if GridWorld.is_terminal(s):
                continue

            old_v = V[s]
            a = pi[s]

            new_v = 0.0
            for s_next, p in GridWorld.get_transition_probs(s, a).items():
                r = GridWorld.reward(s, s_next)
                new_v += p * (r + gamma * V[s_next])

            V[s] = new_v
            delta = max(delta, abs(old_v - new_v))

        if delta < theta:
            break

    return V


def policy_improvement(
    old_pi: Dict[State, Action],
    V: Dict[State, float],
    gamma: float = 0.9
):
    new_pi: Dict[State, Action] = {}
    policy_stable = True

    for s in all_states():
        if GridWorld.is_terminal(s):
            continue

        old_a = old_pi[s]
        best_a = None
        best_q = float("-inf")

        for a in GridWorld.ACTIONS:
            q_sa = 0.0
            for s_next, p in GridWorld.get_transition_probs(s, a).items():
                r = GridWorld.reward(s, s_next)
                q_sa += p * (r + gamma * V[s_next])

            if q_sa > best_q:
                best_q = q_sa
                best_a = a

        new_pi[s] = best_a

        if best_a != old_a:
            policy_stable = False

    return new_pi, policy_stable


def policy_iteration(
    gamma: float = 0.9,
    theta: float = 1e-4
):
    # Initial policy: always move right
    pi: Dict[State, Action] = {}
    for s in all_states():
        if not GridWorld.is_terminal(s):
            pi[s] = 1

    while True:
        V = policy_evaluation(pi, gamma, theta)
        new_pi, stable = policy_improvement(pi, V, gamma)

        if stable:
            return V, new_pi

        pi = new_pi


if __name__ == "__main__":
    V_pi, pi_pi = policy_iteration()

    print("Policy Iteration Value Function:")
    for y in reversed(range(GridWorld.HEIGHT)):
        row = []
        for x in range(GridWorld.WIDTH):
            s = (x, y)
            if s in GridWorld.OBSTACLES:
                row.append(" ##### ")
            else:
                row.append("{:7.2f}".format(V_pi.get(s, 0.0)))
        print(" ".join(row))

    print("\nPolicy Iteration Policy:")
    print_policy(pi_pi)