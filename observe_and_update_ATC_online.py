import pandas as pd
import numpy as np

import os, sys, time

from online_update import OnlineUpdateMoD
from observe import InThorMagniDataset


def test_online_on_atc(decay_rate):
    time_list = []
    
    onlineUpdateMoD = OnlineUpdateMoD(
        decay_rate=decay_rate,
        config_file="config_atc_b2.yaml",
        current_cliff=None,
        output_cliff_folder=f"coral_atc_try3/{decay_rate}",
        save_fig_folder=f"save_fig_online")
    
    ## parameters if use a cone view
    # robot_position = np.array([0, 0])  # robot view location, 2d
    # robot_facing_angle = 0  # robot facing direction (radians), 0 means facing right along the x-axis
    # fov_angle = np.radians(120)  # Field of view angle in radians
    # fov_radius = 2  # Field of view radius
    
    # observe_period = 1000  # Observation period
    # observe_start_time = 0  # Observation start time

    ##### For cone-shaped field of view #####
    # observed_traj = thor_magni.get_observed_traj(robot_position, robot_facing_angle, fov_angle, fov_radius, observe_start_time, observe_period)    
    # print(observed_traj)
    # thor_magni.plot_fov(robot_position, robot_facing_angle, fov_angle, fov_radius, observed_traj, f"cov")
    
    
    #### For ATC rectangular region ####
    # obs_x = 29
    # obs_y = -19
    # delta_x = 3
    # delta_y = 3

    for hour in range(9, 21, 1):
        b_time = time.time()
            
        # for batch_num in range(1, 6):
        thor_magni = InThorMagniDataset('config_atc_b2.yaml', raw_data_file=f'atc-1s-ds-1024-split-hour/split_data_random/atc-1024-{hour}_train.csv')
        # thor_magni = InThorMagniDataset('config_atc_b2.yaml', raw_data_file=f'atc-1s-ds-1024-split-hour/online_train/{hour}/{hour}_b{batch_num}.csv')
        # observed_traj = thor_magni.get_observed_traj_region_all_time(obs_x, obs_y, delta_x, delta_y)
        observed_traj = thor_magni.get_observed_traj_all_area_all_time()
        # thor_magni.plot_region_atc(obs_x, obs_y, delta_x, delta_y, observed_traj, f"{hour}_{hour + 1}")
        # onlineUpdateMoD.updateMoD(observed_traj, f"ATC1024_{hour}_{hour + 1}_b{batch_num}")
        onlineUpdateMoD.updateMoD(observed_traj, f"ATC1024_{hour}_{hour + 1}_online")
        
        b_end_time = time.time()
        batch_time = b_end_time - b_time
        time_list.append(batch_time)
        print(f"Time elapsed: {batch_time}")
        
    return time_list


# decay_rate = float(sys.argv[1])
decay_rate = 0.5
time_list = test_online_on_atc(decay_rate=decay_rate)

## save time to a file
os.makedirs(f"online_mod_res_atc/runtime", exist_ok=True)
with open(f"online_mod_res_atc/runtime/online_train.txt", "w") as f:
    for batch_time in time_list:
        f.write(f"Time elapsed: {batch_time}\n")
    f.write(f"Total time: {sum(time_list)}")
    f.write(f"Average time: {sum(time_list)/len(time_list)}")

