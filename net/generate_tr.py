import os
import sys
import urllib.request

sumo_home = r"C:\Users\FX506HE\sumo-traffic-rl\.venv\Lib\site-packages\sumo"
os.environ["SUMO_HOME"] = sumo_home
sys.path.append(os.path.join(sumo_home, "tools"))

import randomTrips

net_path = os.path.abspath("grid.net.xml")
rou_path = os.path.abspath("grid.rou.xml")

sys.argv = [
    "randomTrips.py",
    "-n", net_path,
    "-r", rou_path,
    "-e", "1000",
    "-p", "0.9"
]

randomTrips.main(randomTrips.get_options())
print("Success")