import random
from typing import Dict, Tuple, List

# Support both package-style and direct execution
try:  # pragma: no cover - import convenience
    from .gridworld import GridWorld, all_states, print_policy
except ImportError:  # when run as a plain script
    from gridworld import GridWorld, all_states, print_policy

State = Tuple[int, int]
Action = int


def generate_episode(env: GridWorld, policy: Dict[State, Action], epsilon: float) -> List[Tuple[State, Action, float]]:
    """
    Generate one episode following epsilon-greedy policy over Q.
    """
    episode: List[Tuple[State, Action, float]] = []
    s = env.reset()
    while True:
        # epsilon-greedy over current policy's action
        if random.random() < epsilon:
            a = random.choice(GridWorld.ACTIONS)
        else:
            a = policy.get(s, 1)
        s_next, r, done = env.step(a)
        episode.append((s, a, r))
        s = s_next
        if done:
            break
    return episode


def monte_carlo_control(
    num_episodes: int = 10000,
    epsilon: float = 0.1,
    gamma: float = 0.9,
) -> Tuple[Dict[Tuple[State, Action], float], Dict[State, Action]]:
    """
    Task 2: Monte Carlo control with epsilon-greedy, tabular Q(s,a).
    """
    env = GridWorld(stochastic=True)

    Q: Dict[Tuple[State, Action], float] = {}
    returns_sum: Dict[Tuple[State, Action], float] = {}
    returns_count: Dict[Tuple[State, Action], int] = {}

    pi: Dict[State, Action] = {}
    for s in all_states():
        if not GridWorld.is_terminal(s):
            pi[s] = random.choice(GridWorld.ACTIONS)

    for _ in range(num_episodes):
        episode = generate_episode(env, pi, epsilon)

        G = 0.0
        visited: set = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r
            sa = (s, a)
            if sa not in visited:
                visited.add(sa)
                returns_sum[sa] = returns_sum.get(sa, 0.0) + G
                returns_count[sa] = returns_count.get(sa, 0) + 1
                Q[sa] = returns_sum[sa] / returns_count[sa]

                # improve policy greedily at state s
                best_a = max(GridWorld.ACTIONS, key=lambda act: Q.get((s, act), 0.0))
                pi[s] = best_a

    return Q, pi


if __name__ == "__main__":
    Q_mc, pi_mc = monte_carlo_control(num_episodes=5000)
    print("Policy learned by Monte Carlo control:")
    print_policy(pi_mc)

