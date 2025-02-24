import pandas as pd
import numpy as np
import time
import os, sys

from online_update import OnlineUpdateMoD
from observe import InThorMagniDataset


def test_online(decay_rate):
    config_file= "config_real.yaml"
    onlineUpdateMoD = OnlineUpdateMoD(
        decay_rate=decay_rate,
        config_file=config_file,
        current_cliff=None,
        output_cliff_folder=f"real_world_exp/cliff_smallgrid/online_{decay_rate}",
        save_fig_folder=f"save_fig_online")
    

    for batch in range(1, 5):
        start_time = time.time()
        thor_magni = InThorMagniDataset(config_file, raw_data_file=f'real_world_exp/train/online/b{batch}.csv')
        observed_traj = thor_magni.get_observed_traj_all_area_all_time()
        onlineUpdateMoD.updateMoD(observed_traj, f"b{batch}")
        end_time = time.time()
        passed_time = end_time - start_time
        with open(f"real_world_exp/runtime/online_time.txt", "a") as f:
            f.write(f"{batch},{passed_time}\n")



    
# decay_rate = float(sys.argv[1])
decay_rate = 0.5

os.makedirs(f"real_world_exp/runtime", exist_ok=True)
with open(f"real_world_exp/runtime/online_time.txt", "w") as f:
    f.write("batch,time\n")

test_online(decay_rate)




