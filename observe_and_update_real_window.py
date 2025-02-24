import pandas as pd
import numpy as np
import time
import os, sys

from online_update_interval import OnlineUpdateMoD
from observe import InThorMagniDataset


def test_online():
    config_file= "config_real.yaml"
    onlineUpdateMoD = OnlineUpdateMoD(
        config_file=config_file,
        current_cliff=None,
        output_cliff_folder=f"real_world_exp/cliff_smallgrid/window",
        save_fig_folder=f"save_fig_online")
    

    for batch in range(1, 5):
        thor_magni = InThorMagniDataset(config_file, raw_data_file=f'real_world_exp/train/window/b{batch}.csv')
        observed_traj = thor_magni.get_observed_traj_all_area_all_time()
        onlineUpdateMoD.updateMoD(observed_traj, f"b{batch}")

start_time = time.time()
test_online()
end_time = time.time()

passed_time = end_time - start_time

os.makedirs(f"real_world_exp/runtime", exist_ok=True)
with open(f"real_world_exp/runtime/window_time.txt", "w") as f:
    f.write(f"Time elapsed: {passed_time}")
print(f"Time elapsed: {passed_time}")