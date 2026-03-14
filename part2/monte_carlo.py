import random
from typing import Dict, Tuple, List

try:
    from .gridworld import GridWorld, all_states, print_policy
except ImportError:
    from gridworld import GridWorld, all_states, print_policy

State = Tuple[int, int]
Action = int


def epsilon_greedy_action(
    Q: Dict[Tuple[State, Action], float],
    s: State,
    epsilon: float
) -> Action:
    if random.random() < epsilon:
        return random.choice(GridWorld.ACTIONS)

    return max(GridWorld.ACTIONS, key=lambda a: Q.get((s, a), 0.0))


def generate_episode(
    env: GridWorld,
    Q: Dict[Tuple[State, Action], float],
    epsilon: float,
    max_steps: int = 500
) -> List[Tuple[State, Action, float]]:
    episode: List[Tuple[State, Action, float]] = []

    s = env.reset()

    for _ in range(max_steps):
        a = epsilon_greedy_action(Q, s, epsilon)
        s_next, r, done = env.step(a)
        episode.append((s, a, r))
        s = s_next

        if done:
            break

    return episode


def monte_carlo_control(
    num_episodes: int = 10000,
    epsilon: float = 0.1,
    gamma: float = 0.9
) -> Tuple[Dict[Tuple[State, Action], float], Dict[State, Action]]:
    env = GridWorld(stochastic=True)

    Q: Dict[Tuple[State, Action], float] = {}
    returns_sum: Dict[Tuple[State, Action], float] = {}
    returns_count: Dict[Tuple[State, Action], int] = {}

    # Initialize all Q-values
    for s in all_states():
        if GridWorld.is_terminal(s):
            continue
        for a in GridWorld.ACTIONS:
            Q[(s, a)] = 0.0

    for _ in range(num_episodes):
        episode = generate_episode(env, Q, epsilon)

        G = 0.0
        visited = set()

        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r
            sa = (s, a)

            # First-visit MC
            if sa in visited:
                continue

            visited.add(sa)
            returns_sum[sa] = returns_sum.get(sa, 0.0) + G
            returns_count[sa] = returns_count.get(sa, 0) + 1
            Q[sa] = returns_sum[sa] / returns_count[sa]

    pi: Dict[State, Action] = {}
    for s in all_states():
        if GridWorld.is_terminal(s):
            continue
        pi[s] = max(GridWorld.ACTIONS, key=lambda a: Q.get((s, a), 0.0))

    return Q, pi


if __name__ == "__main__":
    Q_mc, pi_mc = monte_carlo_control(num_episodes=10000, epsilon=0.1, gamma=0.9)
    print("Monte Carlo Learned Policy:")
    print_policy(pi_mc)