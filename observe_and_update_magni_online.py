import pandas as pd
import numpy as np

import os, sys
import time

# from online_update import OnlineUpdateMoD
from online_update import OnlineUpdateMoD
from observe import InThorMagniDataset



def test_online_on_thor_magni():
    
    onlineUpdateMoD = OnlineUpdateMoD(
        config_file="config_magni.yaml",
        current_cliff=None,
        output_cliff_folder=f"online_mod_res_magni_A_first_online",
        save_fig_folder=f"save_fig_online")
    
    ## parameters if use a cone view
    # robot_position = np.array([0, 0])  # robot view location, 2d
    # robot_facing_angle = 0  # robot facing direction (radians), 0 means facing right along the x-axis
    # fov_angle = np.radians(120)  # Field of view angle in radians
    # fov_radius = 2  # Field of view radius
    
    observe_period = 200  # Observation period
    observe_start_time = 0  # Observation start time

    ##### For cone-shaped field of view #####
    # observed_traj = thor_magni.get_observed_traj(robot_position, robot_facing_angle, fov_angle, fov_radius, observe_start_time, observe_period)    
    # print(observed_traj)
    # thor_magni.plot_fov(robot_position, robot_facing_angle, fov_angle, fov_radius, observed_traj, f"cov")
    
    
    # #### For rectangular region ####
    obs_x = 1
    obs_y = 1
    delta_x = 6
    delta_y = 3
    
    # # obs_x = -7
    # # obs_y = -3.5
    # # delta_x = 5
    # # delta_y = 5
    
    for exp_type in ['A', 'B']:
        thor_magni = InThorMagniDataset('config_magni.yaml', raw_data_file=f'thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/{exp_type}.csv')
        
        observe_start_time = 0
        for observe_start_time in range(0, 10, 1):
            observe_start_time = observe_start_time * observe_period
            print("-------------------- In time period -------------------")
            print(observe_start_time, observe_start_time + observe_period)
            # observed_traj = thor_magni.get_observed_traj_region(obs_x, obs_y, delta_x, delta_y, observe_start_time, observe_period)
            observed_traj = thor_magni.get_observed_traj_all_area(obs_x, obs_y, delta_x, delta_y, observe_start_time, observe_period)
            
            # thor_magni.plot_region(obs_x, obs_y, delta_x, delta_y, observed_traj, f"{observe_start_time}_{observe_start_time + observe_period}")
            onlineUpdateMoD.updateMoD(observed_traj, f"{exp_type}_{observe_start_time}_{observe_start_time + observe_period}")


def test_online_on_thor_magni_split(batch_num):
    
    onlineUpdateMoD = OnlineUpdateMoD(
        config_file="config_magni.yaml",
        current_cliff=None,
        output_cliff_folder=f"online_mod_res_magni_A_first_split_online",
        save_fig_folder=f"save_fig_online")
    
    observe_period = 200  # Observation period
    observe_start_time = 0  # Observation start time

    # #### For rectangular region ####
    obs_x = 1
    obs_y = 1
    delta_x = 6
    delta_y = 3
    
    for exp_type in ['A', 'B']:
        for observe_start_time_ind in range(0, 10, 1):
            # for i in range(10):
            observe_start_time = observe_start_time_ind * observe_period
            print("-------------------- In time period -------------------")
            print(observe_start_time, observe_start_time + observe_period)
            raw_data_file = f'thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/split_data/{exp_type}_train_{observe_start_time}_{observe_start_time + observe_period}_{batch_num}.csv'
            thor_magni = InThorMagniDataset('config_magni.yaml', raw_data_file=raw_data_file)
            # observed_traj = thor_magni.get_observed_traj_region(obs_x, obs_y, delta_x, delta_y, observe_start_time, observe_period)
            observed_traj = thor_magni.get_observed_traj_all_area(obs_x, obs_y, delta_x, delta_y, observe_start_time, observe_period)
            # observed_traj = thor_magni.get_observed_traj_all_area_all_time()
            
            # thor_magni.plot_region(obs_x, obs_y, delta_x, delta_y, observed_traj, f"{observe_start_time}_{observe_start_time + observe_period}")
            onlineUpdateMoD.updateMoD(observed_traj, f"{exp_type}_{observe_start_time}_{observe_start_time + observe_period}")


def test_online_on_thor_magni_split_random(exp_first='A', model_type='online', decay_rate=0.9):
    # model_type = 'online'
    
    onlineUpdateMoD = OnlineUpdateMoD(
        decay_rate=decay_rate,
        config_file="config_magni.yaml",
        current_cliff=None,
        output_cliff_folder=f"online_mod_res_magni_{exp_first}_first_split_random_{model_type}_decay_{decay_rate}",
        save_fig_folder=f"save_fig_online")
    
    observe_period = 200  # Observation period
    observe_start_time = 0  # Observation start time
    
    if exp_first == 'A':
        exp_types = ['A', 'B']
    elif exp_first == 'B':
        exp_types = ['B', 'A']
        
    for exp_type in exp_types:
        for observe_start_time_ind in range(0, 10, 1):
            start_time = time.time()
            print(start_time)
            observe_start_time = observe_start_time_ind * observe_period
            print("-------------------- In time period -------------------")
            print(exp_type, observe_start_time, observe_start_time + observe_period)
            raw_data_file = f'thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/split_data_random/{exp_type}_train_{observe_start_time}_{observe_start_time + observe_period}.csv'
            thor_magni = InThorMagniDataset('config_magni.yaml', raw_data_file=raw_data_file)
            # observed_traj = thor_magni.get_observed_traj_region(obs_x, obs_y, delta_x, delta_y, observe_start_time, observe_period)
            # observed_traj = thor_magni.get_observed_traj_all_area(obs_x, obs_y, delta_x, delta_y, observe_start_time, observe_period)
            observed_traj = thor_magni.get_observed_traj_all_area_all_time()
            
            # thor_magni.plot_region(obs_x, obs_y, delta_x, delta_y, observed_traj, f"{observe_start_time}_{observe_start_time + observe_period}")
            onlineUpdateMoD.updateMoD(observed_traj, f"{exp_type}_{observe_start_time}_{observe_start_time + observe_period}")

            end_time = time.time()
            print(end_time, end_time - start_time)
            with open(f"online_mod_res_magni_{exp_first}_first_split_random_{model_type}_decay_{decay_rate}/timelog.log", 'a') as log_file:
                log_file.write(f"{end_time - start_time:.2f}\n")


############ To train on split, 10 fold ############
# batch_num = sys.argv[1]
# batch_num = 1
# test_online_on_thor_magni_split(batch_num)

decay_rate = float(sys.argv[1])
############ To train on random sample 10% split, 1 fold ############
test_online_on_thor_magni_split_random(exp_first='A', model_type='online', decay_rate=decay_rate)