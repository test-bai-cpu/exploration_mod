import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

import os

import compute_NLL_utils as utils


def compute_for_magni(model_type, threshold, decay_rate=None):
    dataset_name = "MAGNI"
    exp_types = ['A', 'B']
    exp_first = 'A'
    nlls = []
    
    for exp_type in exp_types:
    # for exp_type in ["A"]:
        for observe_start_time_ind in range(0, 10, 1):
        # for observe_start_time_ind in range(0, 1, 1):
            observe_start_time = observe_start_time_ind * 200
            # print("-------------------- In time period -------------------")
            print(exp_type, observe_start_time, observe_start_time + 200)
            test_data_file = f'thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/split_data_random/{exp_type}_test_{observe_start_time}_{observe_start_time + 200}.csv'
            
            MoD_data = None
            if model_type in ["all"]:
                MoD_file = f"online_mod_res_magni_{exp_first}_first_split_random_{model_type}/{exp_type}_{observe_start_time}_{observe_start_time + 200}_{model_type}.csv"
                MoD_data = utils.read_MoD_data_velocity(MoD_file)
            elif model_type in ["online"]:
                if decay_rate is not None:
                    MoD_file = f"online_mod_res_magni_{exp_first}_first_split_random_{model_type}_decay_{decay_rate}/{exp_type}_{observe_start_time}_{observe_start_time + 200}_{model_type}.csv"
                    MoD_data = utils.read_MoD_data_velocity(MoD_file)
                else:
                    MoD_file = f"online_mod_res_magni_{exp_first}_first_split_random_{model_type}_decay_0.5_v3/{exp_type}_{observe_start_time}_{observe_start_time + 200}_{model_type}.csv"
                    MoD_data = utils.read_MoD_data_decay_version_magni(MoD_file)
            elif model_type in ["window"]:
                MoD_file = f"online_mod_res_magni_{exp_first}_first_split_random_{model_type}/{exp_type}_{observe_start_time}_{observe_start_time + 200}_all.csv"
                # MoD_data = utils.read_MoD_data_velocity(MoD_file)
                MoD_data = utils.read_MoD_data_motionangle_magni(MoD_file)
            elif model_type in ["interval"]:
                MoD_file = f"online_mod_res_magni_{exp_first}_first_split_random_{model_type}/{exp_type}_{observe_start_time}_{observe_start_time + 200}_{model_type}.csv"
                MoD_data = utils.read_MoD_data_velocity(MoD_file)
                
            test_data = utils.read_test_data(datafile=test_data_file, dataset=dataset_name)
            
            each_nlls, not_find, average_nll = utils.compute_nll(test_data, MoD_data, threshold)
            
            print(np.mean(each_nlls), not_find)
            if each_nlls is not None:
                nlls += each_nlls

            else:
                print(f"File {MoD_file} not found")
            
    average_nll = np.mean(nlls)
    print("For model type: ", model_type, " NLL:", average_nll, "not find: ", not_find)
    
    return average_nll, not_find


def compute_for_magni_new(model_type, threshold, decay_rate=None):
    dataset_name = "MAGNIv2"
    nlls = []
    
    exp_type = "B"
    batch_num = 5
    
    test_data_file = f'magni-res-v2/B_test.csv'
    # test_data_file = f'magni-res-v2/A_test.csv'
    # test_data_file = f'magni-res-v2/online_train/B_b5.csv'
    
    MoD_data = None
    if model_type in ["all"]:
        MoD_file = f"magni-res-v2/cliff/{model_type}_train/{exp_type}_b{batch_num}_cliff.csv"
        MoD_data = utils.read_MoD_data_motionangle_magni(MoD_file)
    elif model_type in ["online"]:
        MoD_file = f"magni-res-v2/cliff/{model_type}_res/try5/decay_{decay_rate}/{exp_type}_b{batch_num}_online.csv"
        # MoD_file = f"magni-res-v2/cliff/{model_type}_train/decay_{decay_rate}/{exp_type}_b{batch_num}_online.csv"
        MoD_data = utils.read_MoD_data_decay_version_magni(MoD_file)
    elif model_type in ["window"]:
        MoD_file = f"magni-res-v2/cliff/{model_type}_train/{exp_type}_b{batch_num}_cliff.csv"
        MoD_data = utils.read_MoD_data_motionangle_magni(MoD_file)
    elif model_type in ["interval"]:
        MoD_file = f"magni-res-v2/cliff/{model_type}_train/{exp_type}_b{batch_num}_cliff.csv"
        MoD_data = utils.read_MoD_data_motionangle_magni(MoD_file)
        
    test_data = utils.read_test_data(datafile=test_data_file, dataset=dataset_name)
    
    each_nlls, not_find, average_nll = utils.compute_nll(test_data, MoD_data, threshold)
    
    print(np.mean(each_nlls), not_find)
    if each_nlls is not None:
        nlls += each_nlls

    else:
        print(f"File {MoD_file} not found")
            
    average_nll = np.mean(nlls)
    print("For model type: ", model_type, " NLL:", average_nll, "not find: ", not_find, "std: ", np.std(nlls))
    
    return average_nll, not_find, np.std(nlls)

######################### MAGNI #########################

# os.makedirs('magni-res-v2', exist_ok=True)
threshold = 0.5

# os.makedirs('magni-res-v2/try5', exist_ok=True)

# for decay_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
for decay_rate in [0.8]:
    for model_type in ["all", "online"]:
    # for model_type in ["interval"]:
    # for model_type in ["online"]:
        print(f"Model type: {model_type}")
        # average_nll, not_find = compute_for_magni(model_type, threshold, decay_rate=decay_rate)
        average_nll, not_find, std_nlls = compute_for_magni_new(model_type, threshold, decay_rate)
        # file_name = f"magni-res-v2/try7/{model_type}_try5.txt"
        # # file_name = f"magni-res-v2/{model_type}.txt"
        # with open(file_name, "a") as f:
        #     f.write(f"{decay_rate},{average_nll},{std_nlls},{not_find}\n")

            
            
# decay_rate = 0.9
# for model_type in ["all", "interval", "window"]:
# for model_type in ["all"]:
#     print(f"Model type: {model_type}")
#     # average_nll, not_find = compute_for_magni(model_type, threshold, decay_rate=decay_rate)
#     average_nll, not_find = compute_for_magni_new(model_type, threshold, decay_rate)
#     file_name = f"magni-res-v2/{model_type}_{decay_rate}.txt"
#     with open(file_name, "w") as f:
#         f.write(f"{average_nll}\n")
#         f.write(f"not find: {not_find}\n")


# for decay_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#     print(f"Decay rate: {decay_rate}")
#     exp_first = 'A'    
#     model_type = "online"
#     magni_nll = compute_for_magni_decay(exp_first, model_type, decay_rate)
#     print(f'MAGNI NLL: {magni_nll}') # MAGNI NLL: 1.9112291522773677


# exp_first = 'A'    
# model_type = "online"
# magni_nll = compute_for_magni(exp_first, model_type)
# print(f'MAGNI NLL: {magni_nll}') # MAGNI NLL: 1.9112291522773677

# Average all 
# # no updated: 1.7900360781490905
## With updated: 1.7904131823431282

# exp_first = 'B'    
# model_type = "online"
# magni_nll = compute_for_magni(exp_first, model_type)
# print(f'MAGNI NLL: {magni_nll}') # MAGNI NLL: 1.9112291522773677


# exp_first = 'A'    
# model_type = "window"
# magni_nll = compute_for_magni(exp_first, model_type)
# print(f'MAGNI NLL: {magni_nll}') # MAGNI NLL: 1.6053394064016695

# exp_first = 'A'    
# model_type = "interval"
# magni_nll = compute_for_magni(exp_first, model_type)
# print(f'MAGNI NLL: {magni_nll}') # MAGNI NLL: 2.860189479033391

# exp_first = 'A'    
# model_type = "online"
# magni_nll = compute_for_magni(exp_first, model_type)
# print(f'MAGNI NLL: {magni_nll}') 
# MAGNI NLL: 1.6468594792445566
# 1.6441994954182098 v2
# 1.6447243964280687 v3

# ##### Interval, A, B should be same #####
# exp_first = 'A'    
# model_type = "interval"
# magni_nll = compute_for_magni(exp_first, model_type)
# print(f'MAGNI NLL: {magni_nll}') # MAGNI NLL: 2.9369037469709744

# exp_first = 'B'    
# model_type = "interval"
# magni_nll = compute_for_magni(exp_first, model_type)
# print(f'MAGNI NLL: {magni_nll}') # MAGNI NLL: 2.6741331096792123


# exp_first = 'A'    
# model_type = "all"
# magni_nll = compute_for_magni(exp_first, model_type)
# print(f'MAGNI NLL: {magni_nll}')

# Updated version: 1.413158459267527
# No updated: 1.4131728507020775

# exp_first = 'B'    
# model_type = "all"
# magni_nll = compute_for_magni(exp_first, model_type)
# print(f'MAGNI NLL: {magni_nll}')

##################################################################


# def compute_for_atc(model_type):
#     dataset_name = "ATC"

#     nlls = []
    
#     for hour in range(9, 21, 1):
#         print(f"-------------------- In time period {hour} -------------------")
#         test_data_file = f'atc-1s-ds-1024-split-hour/split_data_random/atc-1024-{hour}_test.csv'
        
#         if model_type in ["all"]:
#             # MoD_file = f"cliff/atc-1024-cliff.csv"
#             MoD_file = "cliff/atc-20121024-ds-full-2cliff.csv"
#             # MoD_file = f"online_mod_res_atc_split_random_{model_type}/ATC1024_{hour}_{hour+1}_{model_type}.csv"
#         else:
#             MoD_file = f"online_mod_res_atc_split_random_{model_type}/ATC1024_{hour}_{hour+1}_{model_type}.csv"
#         # MoD_file = f"/home/yufei/research/for-desktop/cliff/cliff-map-hours/cliff-map-{hour}.csv"
#         # nll, not_find = compute_nll(test_data_file, MoD_file, dataset_name)
#         # if nll is not None:
#         #     nlls.append(nll)
#         # else:
#         #     print(f"File {MoD_file} not found")
        
#         each_nlls, not_find = utils.compute_nll(test_data_file, MoD_file, dataset_name)
#         if each_nlls is not None:
#             nlls += each_nlls
#             # print(nlls)
#         else:
#             print(f"File {MoD_file} not found")
        
            
#     average_nll = np.mean(nlls)
    
#     return average_nll

# def compute_for_atc_hour(model_type, decay_rate):
#     dataset_name = "ATC"

#     nlls = []
    
#     for hour in range(9, 21, 1):
#     # for hour in range(10, 11, 1):
#         print(f"-------------------- In time period {hour} -------------------")
#         test_data_file = f'atc-1s-ds-1024-split-hour/split_data_random/atc-1024-{hour}_test.csv'
        
#         if model_type in ["all"]:
#             # MoD_file = "cliff/atc-20121024-ds-full-2cliff.csv"
#             # MoD_file = f"cliff/atc-1024-cliff.csv"
#             MoD_file = f"run-along/online_mod_res_atc_split_random_{model_type}_single_file_v2/ATC1024_{hour}_{hour+1}_{model_type}.csv"
#             # MoD_file = f"online_mod_res_atc_split_random_{model_type}/ATC1024_{hour}_{hour+1}_{model_type}.csv"
#         elif model_type in ["online"]:
#             # MoD_file = f"online_mod_res_atc_split_random_{model_type}_decay_{decay_rate}_b2_alldecay_debug/ATC1024_{hour}_{hour+1}_{model_type}.csv"
#             MoD_file = f"online_mod_res_atc_split_random_{model_type}_decay_{decay_rate}/ATC1024_{hour}_{hour+1}_{model_type}.csv"
#             print("here")
#         elif model_type in ["interval"]:
#             MoD_file = f"online_mod_res_atc_split_random_{model_type}/ATC1024_{hour}_{hour+1}_{model_type}.csv"
        
#         each_nlls, not_find = utils.compute_nll(test_data_file, MoD_file, dataset_name)
#         if each_nlls is not None:
#             # nlls += each_nlls
#             nlls.append((np.mean(each_nlls), not_find))
#             print("For hour", hour, "NLL:", np.mean(each_nlls), not_find)
#         else:
#             print(f"File {MoD_file} not found")
    
#     return nlls

# def compute_for_atc_decay(model_type, decay_rate):
#     dataset_name = "ATC"

#     nlls = []
    
#     for hour in range(9, 21, 1):
#         # print(f"-------------------- In time period {hour} -------------------")
#         test_data_file = f'atc-1s-ds-1024-split-hour/split_data_random/atc-1024-{hour}_test.csv'
        
#         if model_type in ["all"]:
#             # MoD_file = f"cliff/atc-1024-cliff.csv"
#             # MoD_file = f"online_mod_res_atc_split_random_{model_type}/ATC1024_{hour}_{hour+1}_{model_type}.csv"
#             MoD_file = f"run-along/online_mod_res_atc_split_random_{model_type}_single_file_v2/ATC1024_{hour}_{hour+1}_{model_type}.csv"
#         else:
#             MoD_file = f"online_mod_res_atc_split_random_{model_type}_decay_{decay_rate}/ATC1024_{hour}_{hour+1}_{model_type}.csv"
#         # MoD_file = f"/home/yufei/research/for-desktop/cliff/cliff-map-hours/cliff-map-{hour}.csv"
#         # nll, not_find = compute_nll(test_data_file, MoD_file, dataset_name)
#         # if nll is not None:
#         #     nlls.append(nll)
#         # else:
#         #     print(f"File {MoD_file} not found")
        
#         each_nlls, not_find = utils.compute_nll(test_data_file, MoD_file, dataset_name)
#         if each_nlls is not None:
#             nlls += each_nlls
#             # print(nlls)
#         else:
#             print(f"File {MoD_file} not found")
        
            
#     average_nll = np.mean(nlls)
    
#     return average_nll


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


# decay_rate = 0.5
# atc_nll_hour_list = []
# # for model_type in ["all"]:
# for model_type in ["online"]:
# # for model_type in ["interval"]:
# # for model_type in ["all", "online", "interval"]:
#     atc_nll_hour_list = compute_for_atc_hour(model_type, decay_rate=decay_rate)
#     if model_type in ["all", "interval"]:
#         file_name = f"atc_nll_hour_{model_type}_oricliff.txt"
#     else:
#         file_name = f"atc_nll_hour_{model_type}_decay{decay_rate}_alldecay3.txt"
#     with open(file_name, "w") as f:
#         for nll in atc_nll_hour_list:
#             f.write(f"{nll}\n")
#         total_avg = np.mean([nll[0] for nll in atc_nll_hour_list])
#         f.write(f"Average NLL: {total_avg}\n")
#         print(f"Average NLL: {total_avg}")
##################################################################