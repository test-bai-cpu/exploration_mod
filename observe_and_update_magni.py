import pandas as pd
import numpy as np

import os, sys

from online_update import OnlineUpdateMoD
from observe import InThorMagniDataset



def test_online_on_thor_magni():
    
    onlineUpdateMoD = OnlineUpdateMoD(
        config_file="config.yaml",
        current_cliff=None,
        output_cliff_folder=f"online_mod_res_magni",
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
    
    
    #### For ATC rectangular region ####
    # obs_x = 29
    # obs_y = -19
    # delta_x = 3
    # delta_y = 3
    

    thor_magni = InThorMagniDataset('config.yaml', raw_data_file='thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/A.csv')
    
    for observe_start_time in range(0, 10, 1):
        observe_start_time = observe_start_time * observe_period
        print("-------------------- In time period -------------------")
        print(observe_start_time, observe_start_time + observe_period)
        # observed_traj = thor_magni.get_observed_traj_region(obs_x, obs_y, delta_x, delta_y, observe_start_time, observe_period)
        observed_traj = thor_magni.get_observed_traj_all_area(obs_x, obs_y, delta_x, delta_y, observe_start_time, observe_period)
        
        # thor_magni.plot_region(obs_x, obs_y, delta_x, delta_y, observed_traj, f"{observe_start_time}_{observe_start_time + observe_period}")
        onlineUpdateMoD.updateMoD(observed_traj, f"A_{observe_start_time}_{observe_start_time + observe_period}")
    

    thor_magni = InThorMagniDataset('config.yaml', raw_data_file='thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/B.csv')
    
    observe_start_time = 0  # Observation start time
    for observe_start_time in range(0, 10, 1):
        observe_start_time = observe_start_time * observe_period
        print("-------------------- In time period -------------------")
        print(observe_start_time, observe_start_time + observe_period)
        # observed_traj = thor_magni.get_observed_traj_region(obs_x, obs_y, delta_x, delta_y, observe_start_time, observe_period)
        observed_traj = thor_magni.get_observed_traj_all_area(obs_x, obs_y, delta_x, delta_y, observe_start_time, observe_period)
        
        # thor_magni.plot_region(obs_x, obs_y, delta_x, delta_y, observed_traj, f"{observe_start_time}_{observe_start_time + observe_period}")
        onlineUpdateMoD.updateMoD(observed_traj, f"B_{observe_start_time}_{observe_start_time + observe_period}")

def test_online_on_atc():
    
    onlineUpdateMoD = OnlineUpdateMoD(
        config_file="config.yaml",
        current_cliff=None,
        output_cliff_folder=f"online_mod_res_atc",
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
        thor_magni = InThorMagniDataset('config.yaml', raw_data_file=f'/home/yufei/research/parse_dataset/parse_atc/atc-1s-ds-1024-split-hour/atc-1024-{hour}.csv')
        observed_traj = thor_magni.get_observed_traj_region_all_time(obs_x, obs_y, delta_x, delta_y)
        # observed_traj = thor_magni.get_observed_traj_all_area(obs_x, obs_y, delta_x, delta_y, observe_start_time, observe_period)
        thor_magni.plot_region_atc(obs_x, obs_y, delta_x, delta_y, observed_traj, f"{hour}_{hour + 1}")
        onlineUpdateMoD.updateMoD(observed_traj, f"ATC1024_{hour}_{hour + 1}")


def test_online_on_art():
    
    onlineUpdateMoD = OnlineUpdateMoD(
        config_file="config.yaml",
        current_cliff=None,
        output_cliff_folder=f"online_mod_res_art",
        save_fig_folder=f"save_fig_online")

    for angle in [0, 90, 180, 270]:
        print("-------------------- In angle -------------------", angle)
        thor_magni = InThorMagniDataset('config.yaml', raw_data_file=f'/home/yufei/research/online_mod/online_mod/online_batch_data/artv2/b1_{angle}.csv')
        observed_traj = thor_magni.get_observed_traj_all_area_all_time()
        # thor_magni.plot_region_atc(obs_x, obs_y, delta_x, delta_y, observed_traj, f"{hour}_{hour + 1}")
        onlineUpdateMoD.updateMoD(observed_traj, f"b1_{angle}")



def observe_and_update(observe_start_time, observe_period, observe_region):
    
    onlineUpdateMoD = OnlineUpdateMoD(
        config_file="config.yaml",
        current_cliff=None,
        output_cliff_folder=f"online_mod_res_all_dates_learning_rate_change",
        save_fig_folder=f"save_fig_online")
    
    ## parameters if use a cone view
    # robot_position = np.array([0, 0])  # robot view location, 2d
    # robot_facing_angle = 0  # robot facing direction (radians), 0 means facing right along the x-axis
    # fov_angle = np.radians(120)  # Field of view angle in radians
    # fov_radius = 2  # Field of view radius

    ##### For cone-shaped field of view #####
    # observed_traj = thor_magni.get_observed_traj(robot_position, robot_facing_angle, fov_angle, fov_radius, observe_start_time, observe_period)    
    # print(observed_traj)
    # thor_magni.plot_fov(robot_position, robot_facing_angle, fov_angle, fov_radius, observed_traj, f"cov")
    
    
    #### For rectangular region ####
    obs_x = 1
    obs_y = 1
    delta_x = 6
    delta_y = 3
    
    # obs_x = -7
    # obs_y = -3.5
    # delta_x = 5
    # delta_y = 5
    

    thor_magni = InThorMagniDataset('config.yaml', raw_data_file='thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/A.csv')
    
    for observe_start_time in range(0, 10, 1):
        observe_start_time = observe_start_time * observe_period
        print("-------------------- In time period -------------------")
        print(observe_start_time, observe_start_time + observe_period)
        observed_traj = thor_magni.get_observed_traj_region(obs_x, obs_y, delta_x, delta_y, observe_start_time, observe_period)
        # observed_traj = thor_magni.get_observed_traj_all_area(obs_x, obs_y, delta_x, delta_y, observe_start_time, observe_period)
        
        thor_magni.plot_region(obs_x, obs_y, delta_x, delta_y, observed_traj, f"{observe_start_time}_{observe_start_time + observe_period}")
        onlineUpdateMoD.updateMoD(observed_traj, f"A_{observe_start_time}_{observe_start_time + observe_period}")
    

    thor_magni = InThorMagniDataset('config.yaml', raw_data_file='thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/B.csv')
    
    observe_start_time = 0  # Observation start time
    for observe_start_time in range(0, 10, 1):
        observe_start_time = observe_start_time * observe_period
        print("-------------------- In time period -------------------")
        print(observe_start_time, observe_start_time + observe_period)
        observed_traj = thor_magni.get_observed_traj_region(obs_x, obs_y, delta_x, delta_y, observe_start_time, observe_period)
        # observed_traj = thor_magni.get_observed_traj_all_area(obs_x, obs_y, delta_x, delta_y, observe_start_time, observe_period)
        
        # thor_magni.plot_region(obs_x, obs_y, delta_x, delta_y, observed_traj, f"{observe_start_time}_{observe_start_time + observe_period}")
        onlineUpdateMoD.updateMoD(observed_traj, f"B_{observe_start_time}_{observe_start_time + observe_period}")
    


test_online_on_thor_magni()
# test_online_on_atc()
# test_online_on_art()