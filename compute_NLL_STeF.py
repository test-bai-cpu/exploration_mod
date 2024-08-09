import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal

import os

def is_valid_covariance(cov):
    # if cov[0, 0] <= 0.02 or cov[1, 1] <= 0.02:
    #     return False
    
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
    


def read_test_data(datafile, dataset="ATC"):
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
    
    MoD['weight'] = MoD.groupby(['x', 'y'])['weight'].transform(lambda x: x / x.sum())
    
    return MoD


def read_MoD_data_new_online(datafile):
    if not os.path.exists(datafile):
        return None
    
    # Read GMM data
    MoD = pd.read_csv(datafile, header=None)
    MoD.columns = ["x", "y", "velocity", "motion_angle", "cov1", "cov2", "cov3", "cov4", "weight", "motion_ratio", "decay_rate"]
    
    MoD['weight'] = MoD.groupby(['x', 'y'])['weight'].transform(lambda x: x / x.sum())
    
    return MoD


def read_cliff_map_data_old_version(datafile):
    MoD = pd.read_csv(datafile, header=None)
    # MoD.columns = ["x", "y", "motion_angle", "velocity",
    #                 "cov1", "cov2", "cov3", "cov4", "weight",
    #                 "motion_ratio", "observation_ratio"]
    MoD.columns = ["x", "y", "motion_angle", "velocity",
                    "cov4", "cov2", "cov3", "cov1", "weight",
                    "observation_ratio", "motion_ratio"]

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
            try:
                prob_res_wrap = weight * multivariate_normal.pdf([point['velocity'], point['motion_angle'] + 2 * np.pi * wrap_num], mean, cov, allow_singular=True)
            except:
                prob_res_wrap = 0
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

def normalize_histogram(histogram):
    total = histogram.sum()
    if total == 0:
        return histogram
    return histogram / total

def get_nll_from_stef(cell_in_stefmap, point):
    direction_weight_list = cell_in_stefmap[2:]
    direction_weight_array = np.array(direction_weight_list)
    direction_weight_normalize = normalize_histogram(direction_weight_array)
    
    index = int(round((4 * point["motion_angle"]) / np.pi)) % 8

    prob = direction_weight_normalize[index]
    
    if prob < 1e-9:
        prob = 1e-9
        
    nll = -np.log(prob)

    return nll

def read_stef_full_map_data(STeF_file):
    data = pd.read_csv(STeF_file, header=None)
    data.columns = ["x", "y", "theta_1", "theta_2", "theta_3", "theta_4", "theta_5", "theta_6", "theta_7", "theta_8"]
    return data.to_numpy()

def find_closest_location(locations, point):
    min_dist = float('inf')
    closest_loc = None
    for loc in locations:
        dist = np.sqrt((loc[0] - point['x'])**2 + (loc[1] - point['y'])**2)
        if dist < min_dist:
            min_dist = dist
            closest_loc = loc
            
    if min_dist > 0.5:
        return None
            
    # if closest_loc is None:
    #     print(point, "No closest location found")
    return closest_loc


def find_nearest_cell(point, STeF_data, r_s):
    current_location = np.array([point['x'], point['y']])
    location_array = STeF_data[:,0:2]

    distance_arr = np.sum(np.power(location_array - current_location, 2), axis=1)
    nearest_index = np.where(distance_arr == np.amin(distance_arr))
    nearest_cell = nearest_index[0][0]
    if (np.sqrt(distance_arr[nearest_cell]) > r_s):
        return None

    return STeF_data[nearest_cell]
    

def compute_nll(test_data_file, STeF_file):
    test_data = read_test_data(datafile=test_data_file, dataset="ATC")
    STeF_data = read_stef_full_map_data(STeF_file)
    
    if STeF_data is None:
        return None

    nlls = []

    not_find = 0
    
    for _, point in test_data.iterrows():
        nearest_cell_in_stefmap = find_nearest_cell(point, STeF_data, 0.5)
        if nearest_cell_in_stefmap is None:
            not_find += 1
            # nlls.append(-np.log(1e-9))
            continue
        else:
            nll = get_nll_from_stef(nearest_cell_in_stefmap, point)
            nlls.append(nll)
        
    # average_nll = np.mean(nlls)
    return nlls, not_find



def compute_for_nll_hour():
    nlls = []
    
    for hour in range(9, 21, 1):
        print(f"-------------------- In time period {hour} -------------------")
        test_data_file = f'atc-1s-ds-1024-split-hour/split_data_random/atc-1024-{hour}_test.csv'
    
        STeF_file = f"stef/atc-stef-maps-full-v2/stef-map-{hour}.csv"

        each_nlls, not_find = compute_nll(test_data_file, STeF_file)
        if each_nlls is not None:
            # nlls += each_nlls
            nlls.append((np.mean(each_nlls), not_find))
            print("For hour", hour, "NLL:", np.mean(each_nlls), not_find)
        else:
            print(f"File {STeF_file} not found")
    
    return nlls


nll_hour_list = compute_for_nll_hour()
file_name = f"stef_atc.txt"

with open(file_name, "w") as f:
    for nll in nll_hour_list:
        f.write(f"{nll}\n")
    total_avg = np.mean([nll[0] for nll in nll_hour_list])
    f.write(f"Average NLL: {total_avg}\n")
    print(f"Average NLL: {total_avg}")