import random
from typing import Dict, Tuple

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


def q_learning(
    num_episodes: int = 10000,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    max_steps: int = 500
) -> Dict[Tuple[State, Action], float]:
    env = GridWorld(stochastic=True)
    Q: Dict[Tuple[State, Action], float] = {}

    for s in all_states():
        if GridWorld.is_terminal(s):
            continue
        for a in GridWorld.ACTIONS:
            Q[(s, a)] = 0.0

    for _ in range(num_episodes):
        s = env.reset()

        for _ in range(max_steps):
            a = epsilon_greedy_action(Q, s, epsilon)
            s_next, r, done = env.step(a)

            if done:
                td_target = r
            else:
                best_next_q = max(Q.get((s_next, a2), 0.0) for a2 in GridWorld.ACTIONS)
                td_target = r + gamma * best_next_q

            Q[(s, a)] = Q[(s, a)] + alpha * (td_target - Q[(s, a)])
            s = s_next

            if done:
                break

    return Q


def greedy_policy_from_q(Q: Dict[Tuple[State, Action], float]) -> Dict[State, Action]:
    pi: Dict[State, Action] = {}

    for s in all_states():
        if GridWorld.is_terminal(s):
            continue
        pi[s] = max(GridWorld.ACTIONS, key=lambda a: Q.get((s, a), 0.0))

    return pi


if __name__ == "__main__":
    Q_q = q_learning(num_episodes=10000, alpha=0.1, gamma=0.9, epsilon=0.1)
    pi_q = greedy_policy_from_q(Q_q)

    print("Q-Learning Learned Policy:")
    print_policy(pi_q)