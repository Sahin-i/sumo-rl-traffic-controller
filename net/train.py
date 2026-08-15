import os
os.environ["SUMO_HOME"] = r"C:\Users\FX506HE\sumo-traffic-rl\.venv\Lib\site-packages\sumo"
os.environ["LIBSUMO_AS_TRACI"] = "1"

import random
import numpy as np
import torch
import torch.nn.functional as F

from sumo_rl import SumoEnvironment
from agent import QNetwork, ReplayBuffer

GAMMA = 0.99
BATCH_SIZE = 64
LR = 1e-4
TARGET_SYNC_EVERY = 1000
EPS_START, EPS_END, EPS_DECAY_STEPS = 1.0, 0.05, 25000
MIN_BUFFER_BEFORE_TRAINING = 2000
NUM_EPISODES = 300
EVAL_EVERY = 25
HARD_BRAKE_DROP = 15.0
SAFETY_PENALTY_PER_EVENT = 20.0


def make_env():
    kwargs = dict(
        net_file="grid.net.xml",
        route_file="grid.rou.xml",
        num_seconds=1000,
        delta_time=5,
        single_agent=False,
        use_gui=False,
        reward_fn="queue"
    )
    return SumoEnvironment(**kwargs)


env = make_env()
ts_ids = env.ts_ids
lane_to_ts = {lane: ts for ts in ts_ids for lane in env.traffic_signals[ts].lanes}


def count_hard_braking(env, lane_to_ts, prev_speed):
    counts = {}
    for ts in set(lane_to_ts.values()):
        counts[ts] = 0
        
    current_speed = {}
    for lane, ts in lane_to_ts.items():
        for veh in env.sumo.lane.getLastStepVehicleIDs(lane):
            speed = env.sumo.vehicle.getSpeed(veh)
            current_speed[veh] = speed
            if veh in prev_speed and (prev_speed[veh] - speed) > HARD_BRAKE_DROP:
                counts[ts] += 1
    return counts, current_speed


agents = {}
for ts in ts_ids:
    obs_dim = env.observation_spaces(ts).shape[0]
    num_actions = env.action_spaces(ts).n
    q_net = QNetwork(obs_dim, num_actions)
    target_net = QNetwork(obs_dim, num_actions)
    target_net.load_state_dict(q_net.state_dict())
    agents[ts] = {
        "q_net": q_net,
        "target_net": target_net,
        "optimizer": torch.optim.Adam(q_net.parameters(), lr=LR),
        "buffer": ReplayBuffer(),
    }


def select_action(ts, obs, eps):
    if random.random() < eps:
        return env.action_spaces(ts).sample()
    with torch.no_grad():
        q_values = agents[ts]["q_net"](torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
        return q_values.argmax(dim=1).item()


def train_step(ts):
    a = agents[ts]
    if len(a["buffer"]) < MIN_BUFFER_BEFORE_TRAINING:
        return
    s, ac, r, s2, d = a["buffer"].sample(BATCH_SIZE)

    if not isinstance(ac, torch.Tensor): ac = torch.tensor(ac, dtype=torch.long)
    else: ac = ac.long()
    
    if not isinstance(r, torch.Tensor): r = torch.tensor(r, dtype=torch.float32)
    if not isinstance(d, torch.Tensor): d = torch.tensor(d, dtype=torch.float32)

    if ac.dim() == 1: ac = ac.unsqueeze(1)
    if r.dim() == 1: r = r.unsqueeze(1)
    if d.dim() == 1: d = d.unsqueeze(1)

    q_pred = a["q_net"](s).gather(1, ac)

    with torch.no_grad():
        best_actions = a["q_net"](s2).argmax(dim=1, keepdim=True)
        q_next = a["target_net"](s2).gather(1, best_actions)
        q_target = r + GAMMA * q_next * (1 - d)
        
    loss = F.mse_loss(q_pred, q_target)
    a["optimizer"].zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(a["q_net"].parameters(), max_norm=10)
    a["optimizer"].step()


def evaluate():
    eval_env = make_env()
    lane_map = {lane: ts for ts in ts_ids for lane in eval_env.traffic_signals[ts].lanes}
    obs = eval_env.reset()
    prev_speed = {}
    total_braking = 0
    
    total_system_queue = 0
    step_count = 0
    
    done = {"__all__": False}
    while not done["__all__"]:
        actions = {ts: select_action(ts, obs[ts], eps=0.0) for ts in ts_ids}
        obs, rewards, done, info = eval_env.step(actions)
        counts, prev_speed = count_hard_braking(eval_env, lane_map, prev_speed)
        total_braking += sum(counts.values())
        step_queue = sum(eval_env.sumo.lane.getLastStepHaltingNumber(lane) for lane in lane_map.keys())
        total_system_queue += step_queue
        step_count += 1
        
    eval_env.close()

    avg_queue_length = total_system_queue / step_count if step_count > 0 else 0
    return avg_queue_length, total_braking


global_step = 0
episode_rewards = {ts: [] for ts in ts_ids}
best_score = float("inf")

for episode in range(NUM_EPISODES):
    obs = env.reset()
    total_reward = {ts: 0 for ts in ts_ids}
    prev_speed = {}
    done = {"__all__": False}

    while not done["__all__"]:
        eps = max(EPS_END, EPS_START - global_step / EPS_DECAY_STEPS)
        actions = {ts: select_action(ts, obs[ts], eps) for ts in ts_ids}

        next_obs, rewards, done, info = env.step(actions)
        braking_counts, prev_speed = count_hard_braking(env, lane_to_ts, prev_speed)

        for ts in ts_ids:
            shaped_reward = rewards[ts] - SAFETY_PENALTY_PER_EVENT * braking_counts[ts]
            agents[ts]["buffer"].push(obs[ts], actions[ts], shaped_reward, next_obs[ts], done[ts])
            total_reward[ts] += shaped_reward
            train_step(ts)

        obs = next_obs
        global_step += 1

        if global_step % TARGET_SYNC_EVERY == 0:
            for ts in ts_ids:
                agents[ts]["target_net"].load_state_dict(agents[ts]["q_net"].state_dict())

    for ts in ts_ids:
        episode_rewards[ts].append(total_reward[ts])

    if (episode + 1) % 10 == 0:
        print(f"episode {episode+1}: " + ", ".join(f"{ts}={total_reward[ts]:.2f}" for ts in ts_ids))

    if (episode + 1) % EVAL_EVERY == 0:
        avg_queue, braking_events = evaluate()
        score = avg_queue + 0.5 * braking_events
        print(f"  [eval] episode {episode+1}: avg_queue={avg_queue:.2f} braking_events={braking_events} score={score:.2f} best={best_score:.2f}")
        
        if score < best_score:
            best_score = score
            for ts in ts_ids:
                torch.save(agents[ts]["q_net"].state_dict(), f"q_net_{ts}_safe_best.pt")
            print(f"  new best: {score:.2f}")

env.close()
print("done. best score:", best_score)
