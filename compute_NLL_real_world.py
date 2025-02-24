import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

import os

import compute_NLL_utils as utils


def compute_for_realworld(model_type, threshold, decay_rate=None):
    dataset_name = "REALWORLD"

    test_data_file = f'real_world_exp/data/b4.csv'
    if model_type in ["all"]:
        MoD_file = f"real_world_exp/cliff_smallgrid/{model_type}/b4_{model_type}.csv"
        MoD_data = utils.read_MoD_data_velocity(MoD_file)
    elif model_type in ["online"]:
        MoD_file = MoD_file = f"real_world_exp/cliff_smallgrid/{model_type}_0.5/b4_{model_type}.csv"
        MoD_data = utils.read_MoD_data_decay_version_atc(MoD_file)
    
    elif model_type in ["interval"]:
        MoD_file = f"real_world_exp/cliff_smallgrid/{model_type}/b4_{model_type}.csv"
        MoD_data = utils.read_MoD_data_velocity(MoD_file)
    
    test_data = utils.read_test_data(datafile=test_data_file, dataset=dataset_name) 
    each_nlls, not_find, average_nll = utils.compute_nll(test_data, MoD_data, threshold)
    
    # print(each_nlls)
    
    print("For model type: ", model_type, " NLL:", average_nll, "not find: ", not_find, "std: ", np.std(each_nlls))

    return average_nll, not_find, np.std(each_nlls)



decay_rate = 0.5
os.makedirs('real_world_exp/nll_res', exist_ok=True)
threshold = 0.5

for model_type in ["all", "interval", "online"]:
    print(f"Model type: {model_type}")
    average_nll, not_find, std_nlls = compute_for_realworld(model_type, threshold, decay_rate)

    total_file_name = f"real_world_exp/nll_res/{model_type}.txt"
    with open(total_file_name, "a") as f:
        f.write(f"{average_nll},{std_nlls},{not_find}\n")