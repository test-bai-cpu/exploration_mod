import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

import os

import compute_NLL_utils as utils


def compute_for_atc_hour(model_type, threshold, decay_rate=None):
    dataset_name = "ATC"

    nlls = []
    hour_mean_nlls = []
    hour_not_find = []
    
    
    # 'online_mod_res_atc_split_random_online_decay_None_saveall_v3/ATC1024_9_10_online.csv'
    
    
    for hour in range(9, 21, 1):
        print(f"-------------------- In time period {hour} -------------------")
        # test_data_file = f'atc-1s-ds-1024-split-hour/split_data_random/atc-1024-{hour}_test.csv'
        test_data_file = f"/home/yufei/research/for-desktop/atc-1s-ds/1028.csv"
        
        if model_type in ["all"]:
            MoD_file = f"run-along/online_mod_res_atc_split_random_{model_type}_single_file_v2/ATC1024_{hour}_{hour+1}_{model_type}.csv"
            MoD_data = utils.read_MoD_data_velocity(MoD_file)
        elif model_type in ["online"]:
            if decay_rate is not None:
                # MoD_file = f"online_mod_res_atc_split_random_{model_type}_decay_{decay_rate}/ATC1024_{hour}_{hour+1}_{model_type}.csv"
                MoD_file = f"coral_atc_try2/{decay_rate}/ATC1024_{hour}_{hour+1}_online_online.csv"
                MoD_data = utils.read_MoD_data_decay_version_atc(MoD_file)
                # MoD_data = utils.read_MoD_data_velocity(MoD_file)
            else:
                print("here")
                # MoD_file = f"online_mod_res_atc_split_random_{model_type}_decay_{decay_rate}_b2_alldecay_debug/ATC1024_{hour}_{hour+1}_{model_type}.csv"
                MoD_file = f"online_mod_res_atc_split_random_{model_type}_decay_0.5_v4/ATC1024_{hour}_{hour+1}_{model_type}.csv"
                MoD_data = utils.read_MoD_data_decay_version_atc(MoD_file)
        elif model_type in ["interval"]:
            MoD_file = f"online_mod_res_atc_split_random_{model_type}/ATC1024_{hour}_{hour+1}_{model_type}.csv"
            MoD_data = utils.read_MoD_data_velocity(MoD_file)
        elif model_type in ["window"]:
            MoD_file = f"online_mod_res_atc_split_random_{model_type}2_b2/ATC1024_{hour}_{hour+1}_interval.csv"
            MoD_data = utils.read_MoD_data_velocity(MoD_file)
        
        test_data = utils.read_test_data(datafile=test_data_file, dataset=dataset_name)
        each_nlls, not_find, average_nll = utils.compute_nll(test_data, MoD_data, threshold)
        
        if each_nlls is not None:
            nlls += each_nlls
            hour_mean_nlls.append(average_nll)
            hour_not_find.append(not_find)
        else:
            print(f"File {MoD_file} not found")
        print(f"Average NLL: {average_nll}, not find: {not_find}")
    return np.mean(nlls), hour_mean_nlls, hour_not_find


def compute_for_atc_all(model_type, threshold, decay_rate=None):
    dataset_name = "ATC"

    nlls = []
    hour_mean_nlls = []
    hour_not_find = []
    
    
    # 'online_mod_res_atc_split_random_online_decay_None_saveall_v3/ATC1024_9_10_online.csv'
    
    
    for hour in range(9, 21, 1):
        print(f"-------------------- In time period {hour} -------------------")
        test_data_file = f'atc-1s-ds-1024-split-hour/split_data_random/atc-1024-{hour}_test.csv'
        
        if model_type in ["all"]:
            MoD_file = f"run-along/online_mod_res_atc_split_random_{model_type}_single_file_v2/ATC1024_{hour}_{hour+1}_{model_type}.csv"
            MoD_data = utils.read_MoD_data_velocity(MoD_file)
        elif model_type in ["online"]:
            if decay_rate is not None:
                MoD_file = f"online_mod_res_atc_split_random_{model_type}_decay_{decay_rate}/ATC1024_{hour}_{hour+1}_{model_type}.csv"
                MoD_data = utils.read_MoD_data_velocity(MoD_file)
            else:
                # MoD_file = f"online_mod_res_atc_split_random_{model_type}_decay_{decay_rate}_b2_alldecay_debug/ATC1024_{hour}_{hour+1}_{model_type}.csv"
                MoD_file = f"online_mod_res_atc_split_random_{model_type}_decay_0.5_saveall_v3/ATC1024_{hour}_{hour+1}_{model_type}.csv"
                MoD_data = utils.read_MoD_data_decay_version_atc(MoD_file)
        elif model_type in ["interval"]:
            MoD_file = f"online_mod_res_atc_split_random_{model_type}/ATC1024_{hour}_{hour+1}_{model_type}.csv"
            MoD_data = utils.read_MoD_data_velocity(MoD_file)
        elif model_type in ["window"]:
            MoD_file = f"online_mod_res_atc_split_random_{model_type}2_b2/ATC1024_{hour}_{hour+1}_interval.csv"
            MoD_data = utils.read_MoD_data_velocity(MoD_file)
        
        test_data = utils.read_test_data(datafile=test_data_file, dataset=dataset_name)
        each_nlls, not_find, average_nll = utils.compute_nll(test_data, MoD_data, threshold)
        
        if each_nlls is not None:
            nlls += each_nlls
            hour_mean_nlls.append(average_nll)
            hour_not_find.append(not_find)
        else:
            print(f"File {MoD_file} not found")
    
    return np.mean(nlls), hour_mean_nlls, hour_not_find



decay_rate = 0.5
# os.makedirs('online_mod_res_atc', exist_ok=True)
threshold = 1.0


os.makedirs('online_mod_res_atc/try8', exist_ok=True)

# for model_type in ["online", "interval", "window"]:
# for model_type in ["all", "interval", "window"]:
# for model_type in ["all"]:

# for decay_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
for decay_rate in [0.5]:
    for model_type in ["online"]:
        print(f"Model type: {model_type}")
        # average_nll, not_find = compute_for_magni(model_type, threshold, decay_rate=decay_rate)
        average_nll, hour_mean_nlls, hour_not_find = compute_for_atc_hour(model_type, threshold, decay_rate)
        # average_nll, hour_mean_nlls, hour_not_find = compute_for_atc_hour(model_type, threshold)
        file_name = f"online_mod_res_atc/try8/atc_{model_type}_{decay_rate}.txt"
        with open(file_name, "w") as f:
            for ind in range(len(hour_mean_nlls)):
                f.write(f"-------------------- In time period {ind+9} -------------------\n")
                f.write(f"hour NLL: {hour_mean_nlls[ind]} , not find: {hour_not_find[ind]}\n")
                
            f.write(f"AVG total: {average_nll}\n")


# for decay_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#     print(f"Decay rate: {decay_rate}")
#     for model_type in ["online"]:
#         # print(f"Model type: {model_type}")
#         # average_nll, not_find = compute_for_magni(model_type, threshold, decay_rate=decay_rate)
#         average_nll, hour_mean_nlls, hour_not_find = compute_for_atc_hour(model_type, threshold, decay_rate=decay_rate)
#         file_name = f"online_mod_res_atc/atc_{model_type}_decay_{decay_rate}.txt"
#         with open(file_name, "w") as f:
#             for ind in range(len(hour_mean_nlls)):
#                 f.write(f"-------------------- In time period {ind+9} -------------------\n")
#                 f.write(f"hour NLL: {hour_mean_nlls[ind]} , not find: {hour_not_find[ind]}\n")
                
#             f.write(f"AVG total: {average_nll}\n")

######################### ATC #########################

# model_type = "online"
# atc_nll = compute_for_atc(model_type)
# print(f'ATC NLL: {atc_nll}') # ATC NLL: 1.6306520245669913

# model_type = "interval"
# atc_nll = compute_for_atc(model_type)
# print(f'ATC NLL: {atc_nll}')

# 1.5835608406648927


# model_type = "all"
# atc_nll = compute_for_atc(model_type)
# print(f'ATC NLL: {atc_nll}')

## test on the last hour: ATC NLL: 4.621127079608595
# -------------------- In time period 9 -------------------
# Average NLL: 3.5875041265598164
# -------------------- In time period 10 -------------------
# Average NLL: 5.579176736394359
# -------------------- In time period 11 -------------------
# Average NLL: 3.1031398801647074
# -------------------- In time period 12 -------------------
# Average NLL: 3.420217068245758
# -------------------- In time period 13 -------------------
# Average NLL: 4.161973251639471
# -------------------- In time period 14 -------------------
# Average NLL: 4.2192706159619755
# -------------------- In time period 15 -------------------
# Average NLL: 4.7309326343466145
# -------------------- In time period 16 -------------------
# Average NLL: 4.506300468167214
# -------------------- In time period 17 -------------------
# Average NLL: 4.109004994875129
# -------------------- In time period 18 -------------------
# Average NLL: 4.88066310191486
# -------------------- In time period 19 -------------------
# Average NLL: 3.995521706725652
# -------------------- In time period 20 -------------------
# Average NLL: 4.621127079608595
# ATC NLL: 4.244636609302291



# atc_nll_list = []
# for decay_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
# # for decay_rate in [0.4]:
#     print(f"Decay rate: {decay_rate}")   
#     model_type = "online"
#     atc_nll = compute_for_atc_decay(model_type, decay_rate)
#     print(f'ATC NLL: {atc_nll}')
#     atc_nll_list.append(atc_nll)



##################################################################