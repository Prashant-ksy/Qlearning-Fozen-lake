# Qlearning-Fozen-lake
This project implements the Q-learning algorithm for solving the Frozen Lake environment from Gymnasium. The Frozen Lake environment is a grid-world game where the agent must navigate through icy terrain to reach a goal tile, avoiding holes that lead to failure.

## Reward Systems
Two different reward systems are explored in this implementation:
1. **Goal-based Reward System**: The agent receives a reward of +1 when the goal is reached and 0 otherwise.
2. **Action-penalty Reward System**: The agent is penalized for each step taken, with the episode ending when the goal is reached.

## Findings
Through experimentation and evaluation, it was discovered that the action-penalty reward system led to better performance compared to the goal-based reward system. This finding highlights the importance of designing appropriate reward structures in reinforcement learning tasks.
