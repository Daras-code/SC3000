# SC3000 Lab Assignment

This repository contains the implementation for SC3000 Lab Assignment Part 1 and Part 2.

The project includes:

* Part 1: Shortest path search on the NYC graph
* Part 2: GridWorld Markov Decision Process and Reinforcement Learning

## Requirements

Python 3 is required to run this project.

If your system uses `python` for Python 2, run the program using:

```
python3 main.py
```

Otherwise you can run normally:

```
python main.py
```

## How to run

Run the following command from the project root folder:

```
python main.py
```

or

```
python3 main.py
```

This will execute:

* Part 1 Task 1 — Dijkstra shortest path
* Part 1 Task 2 — UCS with energy constraint
* Part 1 Task 3 — A* with energy constraint
* Part 2 Task 1 — Value Iteration and Policy Iteration
* Part 2 Task 2 — Monte Carlo Control
* Part 2 Task 3 — Q-learning

## Project structure


part1/
    task1_dijkstra.py
    task2_search.py
    task3_astar.py

part2/
    gridworld.py
    value_iteration.py
    policy_iteration.py
    monte_carlo.py
    q_learn.py
    task1_mdp.py

data/
    G.json
    Dist.json
    Cost.json
    Coord.json

main.py
README.md
