import os
os.environ["SUMO_HOME"] = r"C:\Users\FX506HE\sumo-traffic-rl\.venv\Lib\site-packages\sumo"

from sumo_rl import SumoEnvironment
import random

env = SumoEnvironment(
    net_file="grid.net.xml",
    route_file="grid.rou.xml",
    num_seconds=1000,
    delta_time=5,
    single_agent=False,
    use_gui=False,
)

obs = env.reset()
print("agents:", env.ts_ids)

total_rewards = {ts: 0 for ts in env.ts_ids}
done = False
while not done:
    actions = {ts: random.randrange(env.action_spaces(ts).n) for ts in env.ts_ids}
    obs, rewards, dones, info = env.step(actions)
    for ts in env.ts_ids:
        total_rewards[ts] += rewards[ts]
    done = dones["__all__"]

print("total reward per intersection (random policy):", total_rewards)
env.close()