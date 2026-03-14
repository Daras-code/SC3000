from typing import Dict, Tuple

# Support both package-style and direct execution
try:  # pragma: no cover - import convenience
    from .gridworld import GridWorld, all_states, print_policy
except ImportError:  # when run as a plain script
    from gridworld import GridWorld, all_states, print_policy

State = Tuple[int, int]
Action = int


def policy_evaluation(pi: Dict[State, Action], gamma: float = 0.9, theta: float = 1e-4) -> Dict[State, float]:
    V: Dict[State, float] = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if GridWorld.is_terminal(s):
                continue
            v = V[s]
            a = pi[s]
            q_sa = 0.0
            for s_next, p in GridWorld.get_transition_probs(s, a).items():
                r = GridWorld.reward(s, s_next)
                q_sa += p * (r + gamma * V[s_next])
            V[s] = q_sa
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V


def policy_improvement(V: Dict[State, float], gamma: float = 0.9) -> Tuple[Dict[State, Action], bool]:
    policy_stable = True
    pi: Dict[State, Action] = {}
    for s in all_states():
        if GridWorld.is_terminal(s):
            continue
        old_a = pi.get(s, 0)
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
        if best_a != old_a:
            policy_stable = False
    return pi, policy_stable


def policy_iteration(gamma: float = 0.9, theta: float = 1e-4) -> Tuple[Dict[State, float], Dict[State, Action]]:
    # Start from an arbitrary policy (e.g., always Right)
    pi: Dict[State, Action] = {}
    for s in all_states():
        if not GridWorld.is_terminal(s):
            pi[s] = 1  # Right

    while True:
        V = policy_evaluation(pi, gamma, theta)
        pi_new, stable = policy_improvement(V, gamma)
        if stable:
            return V, pi_new
        pi = pi_new


if __name__ == "__main__":
    V_pi, pi_pi = policy_iteration()
    print("Policy Iteration values (sample):")
    for y in reversed(range(GridWorld.HEIGHT)):
        row = []
        for x in range(GridWorld.WIDTH):
            s = (x, y)
            if s in GridWorld.OBSTACLES:
                row.append("  ####  ")
            else:
                row.append("{:7.2f}".format(V_pi.get(s, 0.0)))
        print(" ".join(row))
    print("\nPolicy from policy iteration:")
    print_policy(pi_pi)

