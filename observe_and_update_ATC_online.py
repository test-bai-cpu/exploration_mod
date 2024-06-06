import pandas as pd
import numpy as np

import os, sys

from online_update import OnlineUpdateMoD
from observe import InThorMagniDataset


def test_online_on_atc():
    
    onlineUpdateMoD = OnlineUpdateMoD(
        config_file="config_atc.yaml",
        current_cliff=None,
        output_cliff_folder=f"online_mod_res_atc_split_random_online",
        save_fig_folder=f"save_fig_online")
    
    ## parameters if use a cone view
    # robot_position = np.array([0, 0])  # robot view location, 2d
    # robot_facing_angle = 0  # robot facing direction (radians), 0 means facing right along the x-axis
    # fov_angle = np.radians(120)  # Field of view angle in radians
    # fov_radius = 2  # Field of view radius
    
    observe_period = 1000  # Observation period
    observe_start_time = 0  # Observation start time

    ##### For cone-shaped field of view #####
    # observed_traj = thor_magni.get_observed_traj(robot_position, robot_facing_angle, fov_angle, fov_radius, observe_start_time, observe_period)    
    # print(observed_traj)
    # thor_magni.plot_fov(robot_position, robot_facing_angle, fov_angle, fov_radius, observed_traj, f"cov")
    
    
    #### For ATC rectangular region ####
    obs_x = 29
    obs_y = -19
    delta_x = 3
    delta_y = 3

    for hour in range(9, 21, 1):
        thor_magni = InThorMagniDataset('config_atc.yaml', raw_data_file=f'atc-1s-ds-1024-split-hour/split_data_random/atc-1024-{hour}_train.csv')
        # observed_traj = thor_magni.get_observed_traj_region_all_time(obs_x, obs_y, delta_x, delta_y)
        observed_traj = thor_magni.get_observed_traj_all_area_all_time()
        # thor_magni.plot_region_atc(obs_x, obs_y, delta_x, delta_y, observed_traj, f"{hour}_{hour + 1}")
        onlineUpdateMoD.updateMoD(observed_traj, f"ATC1024_{hour}_{hour + 1}")


test_online_on_atc()