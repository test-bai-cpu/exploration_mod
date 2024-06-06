import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal

import os

def is_valid_covariance(cov):
    # Check if the covariance matrix is valid (non-zero and positive definite)
    try:
        np.linalg.cholesky(cov)
        return True
    except np.linalg.LinAlgError:
        return False
    
def is_too_narrow(cov, threshold=1e-6):
    determinant = np.linalg.det(cov)
    return determinant < threshold

def regularize_covariance(cov, epsilon=1e-6):
    cov += epsilon * np.eye(cov.shape[0])
    return cov
    
def preprocess_gmm_data(gmm_data):
    locations = gmm_data[['x', 'y']].drop_duplicates().values
    gmm_dict = {}
    for loc in locations:
        loc_data = gmm_data[(gmm_data['x'] == loc[0]) & (gmm_data['y'] == loc[1])]
        valid_rows = []
        for _, row in loc_data.iterrows():
            cov = np.array([[row['cov1'], row['cov2']], [row['cov3'], row['cov4']]])
            if row['weight'] > 0 and is_valid_covariance(cov):
                if is_too_narrow(cov):
                    cov = regularize_covariance(cov, epsilon=1e-6)
                row['cov1'], row['cov2'], row['cov3'], row['cov4'] = cov.flatten()
                valid_rows.append(row)
        
        if valid_rows:
            loc_data = pd.DataFrame(valid_rows)
            weights = loc_data['weight'].values
            weights /= weights.sum()  # Normalize the weights
            loc_data['weight'] = weights
            gmm_dict[(loc[0], loc[1])] = loc_data
        
    return gmm_dict

def read_test_data(datafile, dataset="MAGNI"):
    if dataset == "MAGNI":
        test_data = pd.read_csv(datafile)
        test_data = test_data.rename(columns={
            'Time': 'time',
            'ag_id': 'person_id',
            'speed': 'velocity',
            'theta_delta': 'motion_angle'
        })
        test_data['motion_angle'] = np.mod(test_data['motion_angle'], 2 * np.pi)
        test_data = test_data[['time', 'x', 'y', 'velocity', 'motion_angle']]
    elif dataset == "ATC":
        test_data = pd.read_csv(datafile, header=None)
        test_data.columns = ["time", "person_id", "x", "y", "velocity", "motion_angle"]
        test_data['motion_angle'] = np.mod(test_data['motion_angle'], 2 * np.pi)
        test_data = test_data[['time', 'x', 'y', 'velocity', 'motion_angle']]

    return test_data

def read_MoD_data(datafile):
    if not os.path.exists(datafile):
        return None
    
    # Read GMM data
    MoD = pd.read_csv(datafile, header=None)
    MoD.columns = ["x", "y", "velocity", "motion_angle", "cov1", "cov2", "cov3", "cov4", "weight", "motion_ratio"]

    return MoD


def read_cliff_map_data_old_version(datafile):
    MoD = pd.read_csv(datafile, header=None)
    MoD.columns = ["x", "y", "motion_angle", "velocity",
                    "cov1", "cov2", "cov3", "cov4", "weight",
                    "motion_ratio", "observation_ratio"]

    return MoD


def nll_of_point(gmm_components, point):
    prob = 0
    for _, component in gmm_components.iterrows():
        mean = [component['velocity'], component['motion_angle']]
        cov = [[component['cov1'], component['cov2']], [component['cov3'], component['cov4']]]
        weight = component['weight']
        # prob += weight * multivariate_normal.pdf([point['velocity'], point['motion_angle']], mean, cov)
        prob_res = []
        for wrap_num in [-1, 0, 1]:
            prob_res_wrap = weight * multivariate_normal.pdf([point['velocity'], point['motion_angle'] + 2 * np.pi * wrap_num], mean, cov, allow_singular=True)
            prob_res.append(prob_res_wrap)
            prob += prob_res_wrap
    # if prob == 0:
    #     print("------------")
    #     print(point)
    #     print(prob_res)
    #     print(gmm_components)
        
    if prob < 1e-9:
        prob = 1e-9
        
    nll = -np.log(prob)

    return nll

def find_closest_location(locations, point):
    min_dist = float('inf')
    closest_loc = None
    for loc in locations:
        dist = np.sqrt((loc[0] - point['x'])**2 + (loc[1] - point['y'])**2)
        if dist < min_dist:
            min_dist = dist
            closest_loc = loc
            
    # if min_dist > 0.5:
    #     return None
            
    if closest_loc is None:
        print(point, "No closest location found")
    return closest_loc


def compute_nll(test_data_file, MoD_file, dataset_name):
    test_data = read_test_data(datafile=test_data_file, dataset=dataset_name)
    MoD_data = read_MoD_data(MoD_file)
    # MoD_data = read_cliff_map_data_old_version(MoD_file)
    
    if MoD_data is None:
        return None
    
    gmm_dict = preprocess_gmm_data(MoD_data)

    locations = list(gmm_dict.keys())
    nlls = []

    for _, point in test_data.iterrows():
        closest_loc = find_closest_location(locations, point)
        if closest_loc is None:
            continue
        gmm_components = gmm_dict[closest_loc]
        nll = nll_of_point(gmm_components, point)
        nlls.append(nll)
        
    average_nll = np.mean(nlls)
    print(f'Average NLL: {average_nll}')
    
    return nlls


def compute_for_magni(exp_first, model_type):
    dataset_name = "MAGNI"
    exp_types = ['A', 'B']

    nlls = []
    
    for exp_type in exp_types:
        for observe_start_time_ind in range(0, 10, 1):
            observe_start_time = observe_start_time_ind * 200
            # print("-------------------- In time period -------------------")
            print(exp_type, observe_start_time, observe_start_time + 200)
            test_data_file = f'thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/split_data_random/{exp_type}_test_{observe_start_time}_{observe_start_time + 200}.csv'
            
            if model_type in ["all", "online"]:
                MoD_file = f"online_mod_res_magni_{exp_first}_first_split_random_{model_type}/{exp_type}_{observe_start_time}_{observe_start_time + 200}_{model_type}.csv"
            else:
                MoD_file = f"online_mod_res_magni_{exp_first}_first_split_random_{model_type}/{exp_type}_{observe_start_time}_{observe_start_time + 200}_{model_type}.csv"
            
            each_nlls = compute_nll(test_data_file, MoD_file, dataset_name)
            if each_nlls is not None:
                nlls += each_nlls
                # print(nlls)
            else:
                print(f"File {MoD_file} not found")
            
            # nll = compute_nll(test_data_file, MoD_file, dataset_name)
            # if nll is not None:
            #     nlls.append(nll)
            # else:
            #     print(f"File {MoD_file} not found") 
            # print(nll)
            
    average_nll = np.mean(nlls)
    
    return average_nll

def compute_for_atc(model_type):
    dataset_name = "ATC"

    nlls = []
    
    for hour in range(9, 21, 1):
        print(f"-------------------- In time period {hour} -------------------")
        test_data_file = f'atc-1s-ds-1024-split-hour/split_data_random/atc-1024-{hour}_test.csv'
        
        if model_type in ["all"]:
            MoD_file = f"/home/yufei/research/for-desktop/cliff/atc/atc-20121024-ds-full-2cliff.csv"
        else:
            MoD_file = f"online_mod_res_atc_split_random_{model_type}/ATC1024_{hour}_{hour+1}_{model_type}.csv"
        # MoD_file = f"/home/yufei/research/for-desktop/cliff/cliff-map-hours/cliff-map-{hour}.csv"
        # nll = compute_nll(test_data_file, MoD_file, dataset_name)
        # if nll is not None:
        #     nlls.append(nll)
        # else:
        #     print(f"File {MoD_file} not found")
        
        each_nlls = compute_nll(test_data_file, MoD_file, dataset_name)
        if each_nlls is not None:
            nlls += each_nlls
            # print(nlls)
        else:
            print(f"File {MoD_file} not found")
        
            
    average_nll = np.mean(nlls)
    
    return average_nll


######################### ATC #########################

# model_type = "online"
# atc_nll = compute_for_atc(model_type)
# print(f'ATC NLL: {atc_nll}') # ATC NLL: 1.6306520245669913

model_type = "interval"
atc_nll = compute_for_atc(model_type)
print(f'ATC NLL: {atc_nll}')

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




######################### MAGNI #########################

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