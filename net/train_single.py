import os
os.environ["SUMO_HOME"] = r"C:\Users\FX506HE\sumo-traffic-rl\.venv\Lib\site-packages\sumo"

import random
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn.functional as F

from sumo_rl import SumoEnvironment
from agent import QNetwork, ReplayBuffer

GAMMA = 0.99
BATCH_SIZE = 64
LR = 1e-3
TARGET_SYNC_EVERY = 200
EPS_START, EPS_END, EPS_DECAY_STEPS = 1.0, 0.05, 25000
MIN_BUFFER_BEFORE_TRAINING = 200
NUM_EPISODES = 300


def evaluate(q_net):
    """Run one greedy (no-exploration) episode on a fresh eval env,
    return the real average per-vehicle waiting time."""
    eval_env = SumoEnvironment(
        net_file="single.net.xml",
        route_file="single.rou.xml",
        num_seconds=1000,
        delta_time=5,
        single_agent=True,
        use_gui=False,
        additional_sumo_cmd="--tripinfo-output eval_tripinfo.xml",
    )
    obs, info = eval_env.reset()
    done = False
    while not done:
        with torch.no_grad():
            q_values = q_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            action = q_values.argmax(dim=1).item()
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
    eval_env.close()

    tree = ET.parse("eval_tripinfo.xml")
    waiting_times = [float(t.get("waitingTime")) for t in tree.getroot().findall("tripinfo")]
    return sum(waiting_times) / len(waiting_times)


env = SumoEnvironment(
    net_file="single.net.xml",
    route_file="single.rou.xml",
    num_seconds=1000,
    delta_time=5,
    single_agent=True,
    use_gui=False,
)

obs_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

q_net = QNetwork(obs_dim, num_actions)
target_net = QNetwork(obs_dim, num_actions)
target_net.load_state_dict(q_net.state_dict())
optimizer = torch.optim.Adam(q_net.parameters(), lr=LR)
buffer = ReplayBuffer()
best_waiting_time = float("inf")

global_step = 0
episode_rewards = []

for episode in range(NUM_EPISODES):
    obs, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        eps = max(EPS_END, EPS_START - global_step / EPS_DECAY_STEPS)
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                action = q_values.argmax(dim=1).item()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        total_reward += reward
        global_step += 1

        if len(buffer) >= MIN_BUFFER_BEFORE_TRAINING:
            s, a, r, s2, d = buffer.sample(BATCH_SIZE)
            q_pred = q_net(s).gather(1, a)
            with torch.no_grad():
                q_next = target_net(s2).max(dim=1, keepdim=True)[0]
                q_target = r + GAMMA * q_next * (1 - d)
            loss = F.mse_loss(q_pred, q_target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10)
            optimizer.step()

        if global_step % TARGET_SYNC_EVERY == 0:
            target_net.load_state_dict(q_net.state_dict())

    episode_rewards.append(total_reward)

    if (episode + 1) % 10 == 0:
        avg = np.mean(episode_rewards[-10:])
        print(f"episode {episode+1:4d}  avg_reward(last10)={avg:8.2f}  eps={eps:.2f}")

    if (episode + 1) % 25 == 0:
        avg_wait = evaluate(q_net)
        print(f"  [eval] episode {episode+1}: avg_wait={avg_wait:.2f}  best_so_far={best_waiting_time:.2f}")
        if avg_wait < best_waiting_time:
            best_waiting_time = avg_wait
            torch.save(q_net.state_dict(), "q_net_best.pt")
            print(f"  new best: {avg_wait:.2f}")

env.close()
print("\nfinal best waiting time seen during training:", best_waiting_time)
print("episode rewards:", episode_rewards)