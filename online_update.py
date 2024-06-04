import time
import os
import sys
import shutil
import math
import random
from typing import List

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from batch_process_data import BatchDataProcess, GridData
from mean_shift import MeanShift
from expectation_maximization import ExpectationMaximization
import online_utils as utils

from pprint import pprint


class OnlineUpdateMoD:
    
    def __init__(
        self,
        config_file: str,
        current_cliff: str,
        output_cliff_folder: str,
        save_fig_folder: str,
    ) -> None:
        self.config_params = utils.read_config_file(config_file)
        if current_cliff is not None and os.path.exists(current_cliff):
            self.current_cliff = utils.read_cliff_map_data(current_cliff)
        else:
            self.current_cliff = None
        # if os.path.exists(output_cliff_folder) and os.path.isdir(output_cliff_folder):
        #     shutil.rmtree(output_cliff_folder)
        self.cliff_csv_folder = output_cliff_folder
        self.save_fig_folder = save_fig_folder
        self.decay_rate = float(self.config_params["decay_rate"])
        self.combine_thres = float(self.config_params["combine_thres"])
        os.makedirs(self.cliff_csv_folder, exist_ok=True)
        os.makedirs(self.save_fig_folder, exist_ok=True)
        
        self.data_batches = BatchDataProcess(
            radius=float(self.config_params["radius"]),
            step=float(self.config_params["step"]),
            dataset_type=self.config_params["dataset_type"])
        
    def updateMoD(self, new_batch_file, output_file_name):
        change_grid_centers = self.data_batches.process_new_batch(new_batch_file)
        
        print("Processing the data..., this batch has ", len(change_grid_centers), " grids.")

        for _, key in enumerate(change_grid_centers):
            print("Now processing grid:", key)
            data = self.data_batches.grid_data[key]

            if len(data.data) == len(data.new_data):
                print("key: ", key, " has first appear.")
                cliffs, N_cur, S_cur, T_cur = self.build_cliff(key, data)
                data.cliff = cliffs
                print(cliffs)
                data.N_cur = N_cur
                data.S_cur = S_cur
                data.T_cur = T_cur

                utils.save_cliff_csv_rows(f"{self.cliff_csv_folder}/{output_file_name}_online.csv", cliffs)
                # utils.save_cliff_csv_rows(f"{self.cliff_csv_folder}/{output_file_name}_all.csv", cliffs)
                # utils.save_cliff_csv_rows(f"{self.cliff_csv_folder}/{output_file_name}_interval.csv", cliffs)
                
                data.importance_value = len(data.new_data)
            else:
                data.importance_value = data.importance_value * self.decay_rate
                print("key: ", key, " has new data in same grid.")
                learning_rate = len(data.new_data) / (data.importance_value + len(data.new_data))
                results = self.update_cliff(key, data, learning_rate, s_type="sEM")
                
                if results == None:
                    cliffs, _, _, _ = self.build_cliff(key, data, if_build_with_new_data=True)
                    cliffs, N_cur, S_cur, T_cur = self.combine_cliff(key, data, cliffs)
                    print("new data combine: ")
                    print(cliffs)
                    utils.save_cliff_csv_rows(f"{self.cliff_csv_folder}/{output_file_name}_online.csv", cliffs)
                else:
                    cliffs, N_cur, S_cur, T_cur = results
                    print("updated cliffs:")
                    print(cliffs)
                    utils.save_cliff_csv_rows(f"{self.cliff_csv_folder}/{output_file_name}_online.csv", cliffs)
                    
                data.cliff = cliffs
                data.N_cur = N_cur
                data.S_cur = S_cur
                data.T_cur = T_cur
                
                data.importance_value = data.importance_value + len(data.new_data)
                
                # ### to update using all the data before, i.e., build cliff using all new + history data
                # all_cliffs, _, _, _ = self.build_cliff(key, data)
                # print("all cliffs from start: ")
                # print(all_cliffs)
                # utils.save_cliff_csv_rows(f"{self.cliff_csv_folder}/{output_file_name}_all.csv", all_cliffs)
                
                # interval_cliffs, _, _, _ = self.build_cliff(key, data, if_build_with_new_data=True)
                # utils.save_cliff_csv_rows(f"{self.cliff_csv_folder}/{output_file_name}_interval.csv", interval_cliffs)
                
    def combine_cliff(self, key, data, update_cliff):
        before_cliff = data.cliff
        before_p = np.array([row[8] for row in before_cliff])
        update_p = np.array([row[8] for row in update_cliff])
        
        update_count = len(data.new_data)
        total_count = data.importance_value + update_count
        before_count = data.importance_value
        
        before_p_new = before_p / (sum(before_p)) * (before_count/total_count)
        for i, row in enumerate(before_cliff):
            row[8] = before_p_new[i]
        update_p_new = update_p / (sum(update_p)) * (update_count/total_count)
        for i, row in enumerate(update_cliff):
            row[8] = update_p_new[i]
        
        total_cliff = before_cliff + update_cliff
        
        m = np.array([row[2:4] for row in total_cliff])
        c = np.array([[[row[4], row[5]], [row[6], row[7]]] for row in total_cliff])
        p = np.array([row[8] for row in total_cliff])
        wind_num = int(self.config_params["wind_num"])
        wind_k = np.arange(-wind_num, wind_num + 1)
        N_new, S_new, T_new = self.compute_sufficient_statistics(len(total_cliff), wind_k, data.new_data, m, c, p, if_check_sum_r=False)

        return total_cliff, N_new, S_new, T_new
        

    def build_cliff(self, key, data: GridData, if_build_with_new_data=False) -> GridData:
        print("Start building the clusters.")
        mean_shifter = MeanShift(
            grid_data=data, 
            convergence_threshold=float(self.config_params["convergence_thres_ms"]),
            group_distance_tolerance=float(self.config_params["group_distance_tolerance"]),
            cluster_all=bool(self.config_params["cluster_all"]),
            too_few_data_thres=int(self.config_params["too_few_data_thres"]),
            max_iteration=int(self.config_params["max_iter_ms"]),
            if_build_with_new_data=if_build_with_new_data,
        )
        mean_shifter.run_mean_shift()
        
        print("yufei test: ", mean_shifter.cluster_centers)
        
        if len(mean_shifter.cluster_centers) == 0:
            return [], None, None, None
        
        if if_build_with_new_data:
            pruned_data = utils.pruned_data_after_ms(data.new_data, mean_shifter.data_cluster_labels)
        else:
            pruned_data = utils.pruned_data_after_ms(data.data, mean_shifter.data_cluster_labels)
        emv = ExpectationMaximization(
            grid_data=pruned_data,
            cluster_centers=mean_shifter.cluster_centers, 
            cluster_covariance=mean_shifter.covariances, 
            mixing_factors=mean_shifter.mixing_factors,
            wind_num=int(self.config_params["wind_num"]),
            convergence_thres=float(self.config_params["convergence_thres_em"]),
            max_iteration=int(self.config_params["max_iter_em"])
        )

        emv.run_em_algorithm()

        if len(emv.mean) == 0:
            return [], None, None, None

        cliffs = []
        
        for cluster_i in range(len(emv.mean)):
            save_row = [
                key[0], key[1],
                emv.mean[cluster_i,0], emv.mean[cluster_i,1],
                emv.cov[cluster_i,0,0], emv.cov[cluster_i,0,1], emv.cov[cluster_i,1,0], emv.cov[cluster_i,1,1],
                emv.mix[cluster_i,0], data.motion_ratio
            ]

            rounded_save_row = [round(value, 5) if not (value is None) else value for value in save_row]
            cliffs.append(rounded_save_row)

        wind_num = int(self.config_params["wind_num"])
        wind_k = np.arange(-wind_num, wind_num + 1)
        
        N_new, S_new, T_new = self.compute_sufficient_statistics(len(emv.mean), wind_k, data.new_data, emv.mean, emv.cov, emv.mix, if_check_sum_r=False)

        return cliffs, N_new, S_new, T_new


    def regularize_cov_matrix(self, cov_matrix, epsilon=1e-6):
        cov_matrix += np.eye(cov_matrix.shape[0]) * epsilon
        return cov_matrix


    def update_cliff(self, key, data: GridData, learning_rate: float, s_type: str = "sEM") -> List:
        print("Start updating the clusters.")
        before_cliff = data.cliff
        print("Before: ", before_cliff)

        if len(before_cliff) == 0:
            return self.build_cliff(key, data, if_build_with_new_data=True)

        raw_data = np.array(data.new_data)    
        num_observations = len(raw_data)
        
        wind_num = int(self.config_params["wind_num"])
        wind_k = np.arange(-wind_num, wind_num + 1)
        cluster_nums = len(before_cliff)
        
        ## m: [velocity, motion_angle]
        m = np.array([row[2:4] for row in before_cliff])
        c = np.array([[[row[4], row[5]], [row[6], row[7]]] for row in before_cliff])
        p = np.array([row[8] for row in before_cliff])

        N_cur = np.array(data.N_cur)
        S_cur = np.array(data.S_cur)
        T_cur = np.array(data.T_cur)

        results = self.compute_sufficient_statistics(cluster_nums, wind_k, raw_data, m, c, p)
        if results is False:
            return
        else:
            N_new, S_new, T_new = results
        
        N_cur = N_cur + learning_rate * (N_new - N_cur)
        S_cur = S_cur + learning_rate * (S_new - S_cur)
        T_cur = T_cur + learning_rate * (T_new - T_cur)

        ### Update pi
        p = np.ones((cluster_nums)) * (1 / cluster_nums)
        for j in range(cluster_nums):
            p[j] = N_cur[j,:]
        p = p / np.sum(p)
        
        ### Update mu
        m = np.zeros((cluster_nums, 2), dtype=float)
        for j in range(cluster_nums):
            S_k = S_cur[j,:]
            N_k = N_cur[j,:]
            if N_k != 0:
                m[j,:] = np.divide(S_k, N_k)
            else:
                m[j,:] = np.zeros_like(m[j,:])
                
        # m[:,1] = utils.wrap_to_2pi_no_round(m[:,1])

        ### Update cov
        c = np.zeros((cluster_nums,2,2), dtype=float)
        for j in range(cluster_nums):
                # np.tile(m_new[j,:], (num_observations, 1))
            T_k = T_cur[j,:,:]
            N_k = N_cur[j,:]
            mu_k = m[j,:]
                    
            if N_k != 0:
                c[j,:,:] = np.divide(T_k, N_k) - np.outer(mu_k, mu_k)
            else:
                c[j,:,:] = np.zeros_like(c[j,:,:])

        cliffs = []
        for cluster_i in range(len(m)):
            ### change order saving to the same as running code, 
            save_row = [
                key[0], key[1],
                m[cluster_i,0], m[cluster_i,1],
                c[cluster_i,0,0], c[cluster_i,0,1], c[cluster_i,1,0], c[cluster_i,1,1],
                p[cluster_i], data.motion_ratio
            ]

            rounded_save_row = [round(value, 5) if not (value is None) else value for value in save_row]
            # utils.save_cliff_csv(self.cliff_csv_file, rounded_save_row)
            # print("Update with new data: ")
            # print(rounded_save_row)
            cliffs.append(rounded_save_row)
        
        return cliffs, N_cur, S_cur, T_cur
    
    def compute_sufficient_statistics(self, cluster_nums, wind_k, raw_data, m, c, p, if_check_sum_r=True):
        num_observations = len(raw_data)
        raw_data = np.array(raw_data)
        r_batch = np.zeros((cluster_nums, len(wind_k), num_observations), dtype=float)

        for j in range(cluster_nums):
            for k in range(len(wind_k)):
                wrap_num = wind_k[k]
                try:
                    likelihood = np.array([multivariate_normal.pdf(np.array([row[0], row[1] + 2 * np.pi * wrap_num ]), mean=m[j,:], cov=c[j,:,:], allow_singular=True) for row in raw_data]) * p[j]
                except:
                    likelihood = 1e-9
                r_batch[j,k,:] = likelihood
        r_batch[r_batch < np.finfo(float).eps] = 0
        
        if if_check_sum_r:
            if np.sum(r_batch) < self.combine_thres:
                return False
        
        sum_r = np.tile(np.sum(r_batch, axis=(0, 1)), (cluster_nums, len(wind_k), 1))
        r_batch = np.divide(r_batch, sum_r, out=np.zeros_like(r_batch), where=sum_r!=0)
        
        ### N_k
        N_new = np.zeros((cluster_nums, 1), dtype=float)
        for j in range(cluster_nums):
            sum_r_j = np.sum(r_batch[j,:,:])
            N_new[j,:] = sum_r_j / num_observations

        ### S_k
        S_new = np.zeros((cluster_nums, 2), dtype=float)
        for j in range(cluster_nums):
            t = np.zeros((num_observations, 2), dtype=float)
            for k in range(len(wind_k)):
                wrap_num = wind_k[k]
                data_copy = raw_data.copy()
                data_copy[:, 1] += 2 * np.pi * wrap_num
                t += data_copy * np.tile(r_batch[j,k,:].reshape(-1,1), (1, 2))
            S_new[j,:] = np.sum(t, axis=0) / num_observations
        
        ### T_k
        T_new = np.zeros((cluster_nums, 2, 2), dtype=float)
        for j in range(cluster_nums):
            t = np.zeros((num_observations, 2, 2), dtype=float)
            for k in range(len(wind_k)):
                wrap_num = wind_k[k]
                data_copy = raw_data.copy()
                data_copy[:, 1] += 2 * np.pi * wrap_num
                d_mod = data_copy
                t[:,0,0] += d_mod[:,0]**2 * r_batch[j,k,:]
                t[:,1,1] += d_mod[:,1]**2 * r_batch[j,k,:]
                t[:,0,1] += d_mod[:,0] * d_mod[:, 1] * r_batch[j,k,:]
                t[:,1,0] = t[:,0,1]
                
            T_new[j,:,:] = np.sum(t, axis=0) / num_observations
            
        return N_new, S_new, T_new