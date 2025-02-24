import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

import os

import compute_NLL_utils as utils


######################### MAPF benchmark #########################
def compute_for_mapf_batch(model_type, threshold, decay_rate=None):
    dataset_name = "MAPF"
      
    exp = "update"
    batch_num = 10
    test_data_file = f'mapf_corl_rebuttal/all_test.csv'
    if model_type in ["all"]:
        MoD_file = f"online_mod_res_mapf/{model_type}_newdata/for_nll/{exp}_split_b{batch_num}_{model_type}.csv"
        # MoD_file = f"online_mod_res_mapf/{model_type}_v3/{exp}_split_b{batch_num}_{model_type}.csv"
        # MoD_file = f"online_mod_res_mapf/{model_type}_v3/for_mapf/{exp}_split_b{batch_num}_{model_type}.csv"
        MoD_data = utils.read_MoD_data_velocity(MoD_file)
    elif model_type in ["online"]:
        if decay_rate is not None:
            MoD_file = f"online_mod_res_mapf/{model_type}_newdata_decay_{decay_rate}/{exp}_split_b{batch_num}_{model_type}.csv"
        else:
            MoD_file = f"online_mod_res_mapf/{model_type}_newdata/{exp}_split_b{batch_num}_{model_type}.csv"
        MoD_data = utils.read_MoD_data_decay_version(MoD_file)
    
    elif model_type in ["interval"]:
        MoD_file = f"online_mod_res_mapf/{model_type}_newdata/for_nll/{exp}_split_b{batch_num}_{model_type}.csv"
        MoD_data = utils.read_MoD_data_velocity(MoD_file)
    elif model_type in ["window"]:
        MoD_file = f"online_mod_res_mapf/{model_type}_newdata/for_nll/{exp}_split_b{batch_num}_{model_type}.csv"
        MoD_data = utils.read_MoD_data_velocity(MoD_file)
    
    test_data = utils.read_test_data(datafile=test_data_file, dataset=dataset_name) 
    each_nlls, not_find, average_nll = utils.compute_nll(test_data, MoD_data, threshold)
    
    print("For model type: ", model_type, " NLL:", average_nll, "not find: ", not_find, "std: ", np.std(each_nlls))

    return average_nll, not_find, np.std(each_nlls)

def compute_for_mapf_batch_update(batch_num, model_type, threshold, decay_rate=None):
    dataset_name = "MAPF"
      
    exp = "update"
    batch_num = batch_num
    test_data_file = f'mapf_corl_rebuttal/all_test.csv'
    if model_type in ["all"]:
        # MoD_file = f"online_mod_res_mapf/{model_type}_newdata/for_nll/{exp}_split_b{batch_num}_{model_type}.csv"
        # MoD_file = f"online_mod_res_mapf/{model_type}_v3/{exp}_split_b{batch_num}_{model_type}.csv"
        MoD_file = f"online_mod_res_mapf/check_history_cliff/for_nll/{exp}_split_b{batch_num}_{model_type}.csv"
        # MoD_file = f"online_mod_res_mapf/{model_type}_v3/for_mapf/{exp}_split_b{batch_num}_{model_type}.csv"
        MoD_data = utils.read_MoD_data_velocity(MoD_file)
    elif model_type in ["online"]:
        if decay_rate is not None:
            MoD_file = f"online_mod_res_mapf/{model_type}_newdata_decay_{decay_rate}/{exp}_split_b{batch_num}_{model_type}.csv"
        else:
            MoD_file = f"online_mod_res_mapf/{model_type}_newdata/{exp}_split_b{batch_num}_{model_type}.csv"
        MoD_data = utils.read_MoD_data_decay_version(MoD_file)
    
    elif model_type in ["interval"]:
        MoD_file = f"online_mod_res_mapf/{model_type}_newdata/for_nll/{exp}_split_b{batch_num}_{model_type}.csv"
        MoD_data = utils.read_MoD_data_velocity(MoD_file)
    elif model_type in ["window"]:
        MoD_file = f"online_mod_res_mapf/{model_type}_newdata/for_nll/{exp}_split_b{batch_num}_{model_type}.csv"
        MoD_data = utils.read_MoD_data_velocity(MoD_file)
    
    test_data = utils.read_test_data(datafile=test_data_file, dataset=dataset_name) 
    each_nlls, not_find, average_nll = utils.compute_nll(test_data, MoD_data, threshold)
    
    print("In Update, batch_num ", batch_num, ". For model type: ", model_type, " NLL:", average_nll, "not find: ", not_find, "std: ", np.std(each_nlls))

    return average_nll, not_find, np.std(each_nlls)

def compute_for_mapf_batch_initial(batch_num, model_type, threshold, decay_rate=None):
    dataset_name = "MAPF"
      
    exp = "initial"
    batch_num = batch_num
    test_data_file = f'mapf_corl_rebuttal/initialv2_test.csv'
    if model_type in ["all"]:
        MoD_file = f"online_mod_res_mapf/{model_type}_newdata/for_nll/{exp}_split_b{batch_num}_{model_type}.csv"
        # MoD_file = f"online_mod_res_mapf/{model_type}_v3/{exp}_split_b{batch_num}_{model_type}.csv"
        # MoD_file = f"online_mod_res_mapf/{model_type}_v3/for_mapf/{exp}_split_b{batch_num}_{model_type}.csv"
        MoD_data = utils.read_MoD_data_velocity(MoD_file)
    elif model_type in ["online"]:
        if decay_rate is not None:
            MoD_file = f"online_mod_res_mapf/{model_type}_newdata_decay_{decay_rate}/{exp}_split_b{batch_num}_{model_type}.csv"
        else:
            MoD_file = f"online_mod_res_mapf/{model_type}_newdata/{exp}_split_b{batch_num}_{model_type}.csv"
        MoD_data = utils.read_MoD_data_decay_version(MoD_file)
    
    elif model_type in ["interval"]:
        MoD_file = f"online_mod_res_mapf/{model_type}_newdata/for_nll/{exp}_split_b{batch_num}_{model_type}.csv"
        MoD_data = utils.read_MoD_data_velocity(MoD_file)
    elif model_type in ["window"]:
        MoD_file = f"online_mod_res_mapf/{model_type}_newdata/for_nll/{exp}_split_b{batch_num}_{model_type}.csv"
        MoD_data = utils.read_MoD_data_velocity(MoD_file)
    
    test_data = utils.read_test_data(datafile=test_data_file, dataset=dataset_name) 
    each_nlls, not_find, average_nll = utils.compute_nll(test_data, MoD_data, threshold)
    
    print("In Initial, batch_num ", batch_num, ". For model type: ", model_type, " NLL:", average_nll, "not find: ", not_find, "std: ", np.std(each_nlls))

    return average_nll, not_find, np.std(each_nlls)

decay_rate = 0.5
# threshold = 3
threshold = 3
# for model_type in ["all", "online"]:
# for model_type in ["online"]:
# for model_type in ["all"]:
#     average_nll, not_find = compute_for_mapf_batch(model_type, threshold, decay_rate=decay_rate)
#     # average_nll, not_find = compute_for_mapf_batch(model_type, threshold)
#     file_name = f"online_mod_res_mapf/mapf_benchmark_{model_type}_newdata_thres{threshold}_final6.txt"
#     with open(file_name, "a") as f:
#         f.write(f"{average_nll}\n")
#         f.write(f"not find: {not_find}\n")
##################################################################



# for decay_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
# for decay_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
# for decay_rate in [0.6]:
#     for model_type in ["online"]:
#         print(f"Model type: {model_type}")
#         average_nll, not_find, std_nlls = compute_for_mapf_batch(model_type, threshold, decay_rate)
#         total_file_name = f"online_mod_res_mapf/try7/mapf_benchmark_{model_type}_allbatches.txt"
#         with open(total_file_name, "a") as f:
#             f.write(f"{decay_rate},{average_nll},{std_nlls},{not_find}\n")
            
            
decay_rate = 0.6
model_type = "interval"

for batch_num in range(1,11):
    print(f"Model type: {model_type}")
    average_nll, not_find, std_nlls = compute_for_mapf_batch_update(batch_num, model_type, threshold, decay_rate)
    total_file_name = f"online_mod_res_mapf/try7/mapf_benchmark_{model_type}_batches_update.txt"
    with open(total_file_name, "a") as f:
        f.write(f"{batch_num},{decay_rate},{average_nll},{std_nlls},{not_find}\n")
        

# for batch_num in range(1,11):
#     print(f"Model type: {model_type}")
#     average_nll, not_find, std_nlls = compute_for_mapf_batch_initial(batch_num, model_type, threshold, decay_rate)
#     total_file_name = f"online_mod_res_mapf/try7/mapf_benchmark_{model_type}_batches_initial.txt"
#     with open(total_file_name, "a") as f:
#         f.write(f"{batch_num},{decay_rate},{average_nll},{std_nlls},{not_find}\n")