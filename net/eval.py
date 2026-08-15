import os
import sys

os.environ["SUMO_HOME"] = r"C:\Users\FX506HE\sumo-traffic-rl\.venv\Lib\site-packages\sumo"

import torch
import torch.nn as nn
import numpy as np
from sumo_rl import SumoEnvironment
from agent import QNetwork

NET_FILE = "grid.net.xml"
ROUTE_FILE = "grid_eval.rou.xml"
HARD_BRAKE_DROP = 4.5

def count_hard_braking(env, lane_to_ts, prev_speed):
    counts = {}
    for ts in set(lane_to_ts.values()):
        counts[ts] = 0
        
    current_speed = {}
    try:
        all_veh_ids = env.sumo.vehicle.getIDList()
        for veh in all_veh_ids:
            lane = env.sumo.vehicle.getLaneID(veh)
            if lane in lane_to_ts:
                ts = lane_to_ts[lane]
                speed = env.sumo.vehicle.getSpeed(veh)
                current_speed[veh] = speed
                if veh in prev_speed and (prev_speed[veh] - speed) > HARD_BRAKE_DROP:
                    counts[ts] += 1
    except Exception:
        pass
    return counts, current_speed

def is_simulation_done(step_res):
    if len(step_res) == 5:
        term, trunc = step_res[2], step_res[3]
        d_term = term.get("__all__", all(term.values())) if isinstance(term, dict) else term
        d_trunc = trunc.get("__all__", all(trunc.values())) if isinstance(trunc, dict) else trunc
        return d_term or d_trunc
    else:
        d = step_res[2]
        return d.get("__all__", all(d.values())) if isinstance(d, dict) else d

def run_fixed_time():
    print("\n[1/2] Fixed time 30 sec")
    env = SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        num_seconds=1000,
        delta_time=5,
        single_agent=False,
        use_gui=True
    )
    
    lane_to_ts = {lane: ts for ts, agent in env.traffic_signals.items() for lane in agent.lanes}
    
    res = env.reset()
    obs = res[0] if isinstance(res, tuple) else res
    
    prev_speed = {}
    total_queues = []
    total_waits = []
    total_hard_brakes = 0
    step_count = 0
    done = False
    
    while not done:
        step_count += 1
        actions = {ts: (step_count // 6) % env.action_spaces(ts).n for ts in env.traffic_signals}
        
        step_res = env.step(actions)
        obs = step_res[0]
        done = is_simulation_done(step_res)
        
        queues = [sum(env.traffic_signals[ts].get_lanes_queue()) for ts in env.traffic_signals]
        waits = [sum(env.traffic_signals[ts].get_accumulated_waiting_time_per_lane()) for ts in env.traffic_signals]
        
        total_queues.append(sum(queues))
        total_waits.append(np.mean(waits))
        
        hb_counts, prev_speed = count_hard_braking(env, lane_to_ts, prev_speed)
        total_hard_brakes += sum(hb_counts.values())
        
    env.close()
    return np.max(total_queues), np.mean(total_waits), total_hard_brakes

def run_double_dqn():
    print("\n[2/2] double dqn")
    env = SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        num_seconds=1000,
        delta_time=5,
        single_agent=False,
        use_gui=True 
    )
    
    lane_to_ts = {lane: ts for ts, agent in env.traffic_signals.items() for lane in agent.lanes}
    
    res = env.reset()
    obs = res[0] if isinstance(res, tuple) else res
    
    agents = {}
    for ts in env.traffic_signals:
        s_dim = env.observation_spaces(ts).shape[0]
        a_dim = env.action_spaces(ts).n
        net = QNetwork(s_dim, a_dim)
        
        weight_file = (f"q_net_{ts}_safe_best.pt")
        if os.path.exists(weight_file):
            net.load_state_dict(torch.load(weight_file))
            net.eval()
        else:
            print(f"file was not found for {ts} at '{weight_file}'.")
            
        agents[ts] = net

    prev_speed = {}
    total_queues = []
    total_waits = []
    total_hard_brakes = 0
    done = False
    
    while not done:
        actions = {}
        with torch.no_grad():
            for ts, net in agents.items():
                s_tensor = torch.FloatTensor(obs[ts]).unsqueeze(0)
                actions[ts] = int(torch.argmax(net(s_tensor), dim=1).item())
                
        step_res = env.step(actions)
        obs = step_res[0]
        done = is_simulation_done(step_res)
        
        queues = [sum(env.traffic_signals[ts].get_lanes_queue()) for ts in env.traffic_signals]
        waits = [sum(env.traffic_signals[ts].get_accumulated_waiting_time_per_lane()) for ts in env.traffic_signals]
        
        total_queues.append(sum(queues))
        total_waits.append(np.mean(waits))
        
        hb_counts, prev_speed = count_hard_braking(env, lane_to_ts, prev_speed)
        total_hard_brakes += sum(hb_counts.values())
        
    env.close()
    return np.max(total_queues), np.mean(total_waits), total_hard_brakes

if __name__ == "__main__":
    baseline_queue, baseline_wait, baseline_brakes = run_fixed_time()
    rl_queue, rl_wait, rl_brakes = run_double_dqn()
    
    queue_imp = ((baseline_queue - rl_queue) / max(1e-5, baseline_queue)) * 100
    wait_imp = ((baseline_wait - rl_wait) / max(1e-5, baseline_wait)) * 100
    brake_imp = ((baseline_brakes - rl_brakes) / max(1, baseline_brakes)) * 100
    
    print("\n" + "="*70)
    print("        BENCHMARK EVALUATION RESULTS (UNSEEN TRAFFIC)        ")
    print("="*70)
    print(f"{'Metric':<28} | {'Fixed-Time':<12} | {'Double DQN':<12} | {'Improvement':<12}")
    print("-" * 70)
    print(f"{'Avg Queue Length (cars)':<28} | {baseline_queue:<12.2f} | {rl_queue:<12.2f} | {queue_imp:>+10.1f}%")
    print(f"{'Avg Waiting Time (s)':<28} | {baseline_wait:<12.2f} | {rl_wait:<12.2f} | {wait_imp:>+10.1f}%")
    print(f"{'Hard Braking Events':<28} | {baseline_brakes:<12} | {rl_brakes:<12} | {brake_imp:>+10.1f}%")
    print("="*70)