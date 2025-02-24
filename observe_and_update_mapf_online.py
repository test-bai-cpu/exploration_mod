import pandas as pd
import numpy as np
import time
import os, sys

from online_update import OnlineUpdateMoD
from observe import InThorMagniDataset


def test_online(decay_rate):
    time_list = []
    
    config_file= "config_mapf.yaml"
    onlineUpdateMoD = OnlineUpdateMoD(
        decay_rate=decay_rate,
        config_file=config_file,
        current_cliff=None,
        output_cliff_folder=f"online_mod_res_mapf/interval_forgettime{decay_rate}",
        save_fig_folder=f"save_fig_online")
    
    for version in ["initial", "update"]:
        for batch in range(1, 11):
            b_time = time.time()
            
            if version == "initial":
                thor_magni = InThorMagniDataset(config_file, raw_data_file=f'mapf_corl_rebuttal/initialv2_split/b{batch}.csv')
            else:
                thor_magni = InThorMagniDataset(config_file, raw_data_file=f'mapf_corl_rebuttal/update_split/b{batch}.csv')
            observed_traj = thor_magni.get_observed_traj_all_area_all_time()
            onlineUpdateMoD.updateMoD(observed_traj, f"{version}_split_b{batch}")

            b_end_time = time.time()
            batch_time = b_end_time - b_time
            time_list.append(batch_time)
            print(f"Time elapsed: {batch_time}")
            
    return time_list


decay_rate = float(sys.argv[1])
time_list = test_online(decay_rate)


os.makedirs(f"online_mod_res_mapf/runtime", exist_ok=True)
with open(f"online_mod_res_mapf/runtime/online_train.txt", "w") as f:
    for batch_time in time_list:
        f.write(f"Time elapsed: {batch_time}\n")
    f.write(f"Total time: {sum(time_list)}")
    f.write(f"Average time: {sum(time_list)/len(time_list)}")

