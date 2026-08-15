# Deep Reinforcement Learning for Urban Traffic Control

This project implements a Double Deep Q-Network (DDQN) to dynamically control traffic signals and alleviate urban gridlock. Using the SUMO (Simulation of Urban MObility) traffic simulator and PyTorch, the AI agent learns to minimize intersection wait times and queue lengths while strictly enforcing safety constraints to prevent dangerous hard-braking events.

## 🚀 Key Features

*   **Double DQN Architecture:** Prevents the overestimation bias common in standard Q-learning by separating action selection (Online Network) from action evaluation (Target Network)[cite: 3].
*   **Custom Safety-Aware Reward Function:** The agent's primary goal is to minimize the system-wide queue length, but it receives a severe penalty (`SAFETY_PENALTY_PER_EVENT = 20.0`) if it forces vehicles to drop their speed by more than 15 m/s abruptly[cite: 3].
*   **High-Density Traffic Simulation:** The model is trained and evaluated on heavy, rush-hour style traffic generated with a spawning period of 0.9 seconds[cite: 4, 5].
*   **Unseen Evaluation Environment:** The evaluation phase uses a completely different random seed (`-s 42`) to guarantee the AI is tested on traffic patterns it has never seen during training[cite: 4].
*   **Automated Benchmarking:** Built-in evaluation scripts automatically run both a standard Fixed-Time controller and the DDQN model side-by-side to calculate exact percentage improvements in queue length, waiting time, and safety[cite: 2].

## 🧠 Neural Network Architecture

The agent's brain is a lightweight, high-speed Multi-Layer Perceptron (MLP) built in PyTorch[cite: 1].
*   **Input:** 1D array representing the current state of the intersection (vehicle positions, current phase)[cite: 1, 3].
*   **Hidden Layers:** Two fully connected layers with 128 neurons each, utilizing ReLU activation[cite: 1].
*   **Output:** The predicted Q-values for each possible traffic light phase[cite: 1].
*   **Memory:** A Replay Buffer with a capacity of 100,000 experiences, sampled in batches of 64 to stabilize training[cite: 1, 3].

## 📂 Repository Structure

The core codebase is located within the `net/` directory (or your working directory):

*   `generate_tr.py`: Generates the training traffic routes (`grid.rou.xml`) using SUMO's `randomTrips.py`[cite: 5].
*   `generate_eval_tr.py`: Generates the unseen evaluation traffic routes (`grid_eval.rou.xml`)[cite: 4].
*   `agent.py`: Contains the PyTorch `QNetwork` and the `ReplayBuffer` classes[cite: 1].
*   `train.py`: The main training loop. It runs the simulation for 300 episodes, trains the agent, and saves the best performing weights (`q_net_{ts}_safe_best.pt`)[cite: 3].
*   `eval.py`: The testing script. It runs both the baseline Fixed-Time controller and the trained DDQN controller, then prints a terminal benchmark comparison[cite: 2].

## 🛠️ How to Run

### 1. Install Dependencies
Ensure you have Python installed, then install the required libraries:
```bash
pip install torch numpy sumo-rl
