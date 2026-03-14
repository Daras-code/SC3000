import random
from typing import Tuple, List, Dict

State = Tuple[int, int]
Action = int  # 0: Up, 1: Right, 2: Down, 3: Left


class GridWorld:
    """
    5x5 stochastic grid world as specified in the assignment.

    - States: (x, y) with x,y in {0..4}
    - Start: (0,0), Goal/terminal: (4,4)
    - Obstacles: (2,1), (2,3)
    - Actions: 0=Up,1=Right,2=Down,3=Left
    - Rewards: step -1, reaching goal +10 and episode ends
    - Transition: intended dir prob 0.8, perpendicular dirs 0.1 each.
    """

    WIDTH = 5
    HEIGHT = 5
    START: State = (0, 0)
    GOAL: State = (4, 4)
    OBSTACLES = {(2, 1), (2, 3)}

    ACTIONS = [0, 1, 2, 3]
    ACTION_DELTAS = {
        0: (0, 1),   # Up
        1: (1, 0),   # Right
        2: (0, -1),  # Down
        3: (-1, 0),  # Left
    }
    ACTION_NAMES = {0: "U", 1: "R", 2: "D", 3: "L"}

    def __init__(self, stochastic: bool = True):
        self.stochastic = stochastic
        self.state: State = self.START

    def reset(self) -> State:
        self.state = self.START
        return self.state

    @staticmethod
    def _in_bounds(s: State) -> bool:
        x, y = s
        return 0 <= x < GridWorld.WIDTH and 0 <= y < GridWorld.HEIGHT

    @classmethod
    def is_terminal(cls, s: State) -> bool:
        return s == cls.GOAL

    @classmethod
    def next_state_det(cls, s: State, a: Action) -> State:
        """Deterministic move for a single action (used by transition model)."""
        if cls.is_terminal(s):
            return s
        dx, dy = cls.ACTION_DELTAS[a]
        nx, ny = s[0] + dx, s[1] + dy
        ns = (nx, ny)
        if (not cls._in_bounds(ns)) or (ns in cls.OBSTACLES):
            return s
        return ns

    @classmethod
    def get_transition_probs(cls, s: State, a: Action) -> Dict[State, float]:
        """
        Full transition model P(s'|s,a) for Task 1 (MDP planning).
        """
        if cls.is_terminal(s):
            return {s: 1.0}

        if a == 0:  # Up
            intended, left, right = 0, 3, 1
        elif a == 1:  # Right
            intended, left, right = 1, 0, 2
        elif a == 2:  # Down
            intended, left, right = 2, 1, 3
        else:  # Left
            intended, left, right = 3, 2, 0

        probs = [(0.8, intended), (0.1, left), (0.1, right)]

        result: Dict[State, float] = {}
        for p, aa in probs:
            ns = cls.next_state_det(s, aa)
            result[ns] = result.get(ns, 0.0) + p
        return result

    @staticmethod
    def reward(s: State, s_next: State) -> float:
        if s == GridWorld.GOAL:
            return 0.0
        if s_next == GridWorld.GOAL:
            return 10.0
        return -1.0

    def step(self, a: Action) -> Tuple[State, float, bool]:
        """
        Environment step for RL (Tasks 2 and 3), sampling the stochastic transition.
        """
        s = self.state
        if self.is_terminal(s):
            return s, 0.0, True

        if self.stochastic:
            if a == 0:
                intended, left, right = 0, 3, 1
            elif a == 1:
                intended, left, right = 1, 0, 2
            elif a == 2:
                intended, left, right = 2, 1, 3
            else:
                intended, left, right = 3, 2, 0

            r = random.random()
            if r < 0.8:
                chosen = intended
            elif r < 0.9:
                chosen = left
            else:
                chosen = right

            s_next = self.next_state_det(s, chosen)
        else:
            s_next = self.next_state_det(s, a)

        r = self.reward(s, s_next)
        self.state = s_next
        done = self.is_terminal(s_next)
        return s_next, r, done


def all_states() -> List[State]:
    """Utility to iterate over all non-obstacle states (including start and goal)."""
    sts: List[State] = []
    for x in range(GridWorld.WIDTH):
        for y in range(GridWorld.HEIGHT):
            if (x, y) not in GridWorld.OBSTACLES:
                sts.append((x, y))
    return sts


def print_policy(pi: Dict[State, Action]) -> None:
    for y in reversed(range(GridWorld.HEIGHT)):
        row = []
        for x in range(GridWorld.WIDTH):
            s = (x, y)
            if s in GridWorld.OBSTACLES:
                row.append("#####")
            elif s == GridWorld.GOAL:
                row.append("  G  ")
            elif s == GridWorld.START:
                a = pi.get(s, 1)
                row.append("S-{} ".format(GridWorld.ACTION_NAMES.get(a, "?")))
            else:
                a = pi.get(s, 1)
                row.append("  {}  ".format(GridWorld.ACTION_NAMES.get(a, "?")))
        print(" ".join(row))


if __name__ == "__main__":
    env = GridWorld()
    print("All states:", all_states())

