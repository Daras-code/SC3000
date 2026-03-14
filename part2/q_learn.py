import random
from typing import Dict, Tuple

# Support both package-style and direct execution
try:  # pragma: no cover - import convenience
    from .gridworld import GridWorld, all_states, print_policy
except ImportError:  # when run as a plain script
    from gridworld import GridWorld, all_states, print_policy

State = Tuple[int, int]
Action = int


def q_learning(
    num_episodes: int = 10000,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.1,
) -> Dict[Tuple[State, Action], float]:
    """
    Task 3: Tabular Q-learning with epsilon-greedy exploration.
    """
    env = GridWorld(stochastic=True)
    Q: Dict[Tuple[State, Action], float] = {}

    for _ in range(num_episodes):
        s = env.reset()
        while True:
            # epsilon-greedy action selection
            if random.random() < epsilon:
                a = random.choice(GridWorld.ACTIONS)
            else:
                a = max(GridWorld.ACTIONS, key=lambda act: Q.get((s, act), 0.0))

            s_next, r, done = env.step(a)

            # TD update
            best_next = max(GridWorld.ACTIONS, key=lambda act: Q.get((s_next, act), 0.0))
            td_target = r + gamma * Q.get((s_next, best_next), 0.0) * (0.0 if done else 1.0)
            old = Q.get((s, a), 0.0)
            Q[(s, a)] = old + alpha * (td_target - old)

            s = s_next
            if done:
                break

    return Q


def greedy_policy_from_q(Q: Dict[Tuple[State, Action], float]) -> Dict[State, Action]:
    pi: Dict[State, Action] = {}
    for s in all_states():
        if GridWorld.is_terminal(s):
            continue
        best_a = max(GridWorld.ACTIONS, key=lambda a: Q.get((s, a), 0.0))
        pi[s] = best_a
    return pi


if __name__ == "__main__":
    Q_q = q_learning(num_episodes=10000, alpha=0.1, gamma=0.9, epsilon=0.1)
    pi_q = greedy_policy_from_q(Q_q)
    print("Policy learned by Q-learning:")
    print_policy(pi_q)

