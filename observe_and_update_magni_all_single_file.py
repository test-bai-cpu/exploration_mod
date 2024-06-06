import pandas as pd
import numpy as np

import os, sys
import time

# from online_update import OnlineUpdateMoD
# from online_update_interval import OnlineUpdateMoD
from online_update_all import OnlineUpdateMoD
from observe import InThorMagniDataset


    # # obs_x = -7
    # # obs_y = -3.5
    # # delta_x = 5
    # # delta_y = 5
    

def test_online_on_thor_magni_split_random(exp_first='A', model_type='online', observe_start_time=200):
    onlineUpdateMoD = OnlineUpdateMoD(
        config_file="config_magni.yaml",
        current_cliff=None,
        output_cliff_folder=f"online_mod_res_magni_{exp_first}_first_split_random_{model_type}_combine",
        save_fig_folder=f"save_fig_online")
    
    observe_period = 200  # Observation period


    # obs_x = -7
    # obs_y = -3.5
    # delta_x = 5
    # delta_y = 5
    
    
    raw_data_file = f'thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/combine_for_train_all/{exp_first}_magni_combine_{observe_start_time}_{observe_start_time + observe_period}.csv'
    # raw_data_file = f'thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/combine_for_train_all/testv2.csv'
    thor_magni = InThorMagniDataset('config_magni.yaml', raw_data_file=raw_data_file)
    # observed_traj = thor_magni.get_observed_traj_region(obs_x, obs_y, delta_x, delta_y, observe_start_time, observe_period)
    # observed_traj = thor_magni.get_observed_traj_region_all_time(obs_x, obs_y, delta_x, delta_y)
    observed_traj = thor_magni.get_observed_traj_all_area_all_time()
    
    # thor_magni.plot_region(obs_x, obs_y, delta_x, delta_y, observed_traj, f"{observe_start_time}_{observe_start_time + observe_period}")
    # onlineUpdateMoD.updateMoD(observed_traj, f"{exp_first}_{observe_start_time}_{observe_start_time+200}")
    onlineUpdateMoD.updateMoD(observed_traj, "testab")




start_time = time.time()
############ To train on random sample 10% split, 1 fold ############
test_online_on_thor_magni_split_random(exp_first='A', model_type='all', observe_start_time=200)
end_time = time.time()

print(f"Total processing time: {end_time - start_time:.2f} seconds.")
# with open(f'processing_times_{hour}.log', 'w') as log_file:
#     log_file.write(f"Total processing time: {end_time - start_time:.2f} seconds.\n")