import os
import math
import collections
import yaml
import csv

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import pandas as pd


def millimeter_to_meter(data, columns):
    for col in columns:
        data[col] = data[col].apply(lambda x: x / 1000)
    return data


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

## wrap to -pi to pi
def wrap_to_pi(circular_value):
    return (circular_value + math.pi) % (2 * math.pi) - math.pi


def wrap_to_2pi(circular_value):
    return np.round(np.mod(circular_value,2*np.pi), 3)


def wrap_to_2pi_no_round(circular_value):
    return np.mod(circular_value,2*np.pi)


def circdiff(circular1, circular2):
    return np.abs(np.arctan2(np.sin(circular1 - circular2), np.cos(circular1 - circular2)))


def distance_wrap_2d(p1, p2):
    ad = circdiff(p1[1], p2[1])
    ld = abs(p1[0] - p2[0])
    dist = math.sqrt(ad**2 + ld**2)
    
    return dist


def distance_wrap_2d_vectorized(A, B):
    ad = circdiff(A[:, 1], B[:, 1])  # Angular differences
    ld = np.abs(A[:, 0] - B[:, 0])  # Linear differences
    dist = np.sqrt(ad**2 + ld**2)
    return dist


def gaussian_kernel(x, kernel_bandwidth):
    K_x = (1 / (kernel_bandwidth * np.sqrt(2 * np.pi))) * np.exp(- (x ** 2) / (2 * kernel_bandwidth ** 2))

    return K_x


def print_array(arr):
    with np.printoptions(precision=2, suppress=True):
        print(arr)


def weighted_mean_2d_vec(p, w=None):
    a = p[:, 1]
    le = p[:, 0]
    
    if w is None:
        w = np.ones_like(a)
    else:
        w = np.asarray(w)

    c = np.sum(np.multiply(np.cos(a), w)) / np.sum(w)
    s = np.sum(np.multiply(np.sin(a), w)) / np.sum(w)
    
    cr_m = np.arctan2(s, c)
    
    l_m = np.sum(np.multiply(le, w)) / np.sum(w)
    mean = np.array([l_m, wrap_to_2pi_no_round(cr_m)])
    
    return mean


def mean_cov_2d_vec(p):
    a = p[:, 1]
    le = p[:, 0]

    c = np.mean(np.cos(a))
    s = np.mean(np.sin(a))
    r = np.sqrt(c**2 + s**2)

    cr_m = np.arctan2(s, c)
    
    l_m = np.mean(le)
    
    m = np.array([l_m, wrap_to_2pi_no_round(cr_m)])
    
    std = np.sqrt(-2 * np.log(r))
    v_c = std**2
    
    v_l = np.var(le, ddof=1)
    
    c = 0
    for j in range(len(a)):
        c +=  (le[j] - l_m) * wrap_to_pi(a[j] - cr_m)
    c = c / (len(a) - 1)
    
    if v_c < 0:
        v_c = 0
    c = np.array([[v_l, c], [c, v_c]])
    
    return m, c


def std_of_circular_linear_data(p):
    if len(p) <= 1:
        return 1.0
    
    origin = np.zeros_like(p)
    ad = circdiff(p[:, 1], origin[:, 1])
    ld = np.abs(p[:, 0] - origin[:, 0])
    dist = np.sqrt(ad**2 + ld**2)
    std = np.std(dist, ddof=1)
    
    return std


def check_grid_data(data_df, grid_data_save_path=None):
    data_df.to_csv(grid_data_save_path, index=False)
    print("max vel: ", data_df["velocity"].max())
    
    
def pruned_data_after_ms(data, data_cluster_labels):
    mask = data_cluster_labels != -1
    pruned_data = data[mask]
    
    return pruned_data
    
    
def plot_cliff_over_velocity_single_grid(key_ind, data, mean, save_fig_folder):
    data = np.array(data)
    
    x = data[:,0] * np.cos(data[:,1])
    y = data[:,0] * np.sin(data[:,1])

    mean_x = mean[:, 0] * np.cos(mean[:, 1])
    mean_y = mean[:, 0] * np.sin(mean[:, 1])
    
    colors = data[:, 1]  * 180 / np.pi
    colors = np.append(colors, [0, 360])
    norm = Normalize()
    norm.autoscale(colors)
    colormap = cm.hsv
    
    # Plot the quiver
    plt.clf()
    plt.close('all')
    plt.figure(figsize=(8, 6))
    plt.quiver(np.zeros_like(x), np.zeros_like(y), x, y, color=colormap(norm(colors)), scale=5)
    plt.quiver(np.zeros_like(mean_x), np.zeros_like(mean_y), mean_x, mean_y, color="black", scale=5)
    
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    cbar = plt.colorbar(sm, shrink = 0.5, ticks=[0, 90, 180, 270, 360], fraction=0.05)
    cbar.ax.tick_params(labelsize=10)
    plt.title("Grid " + str(key_ind))
    # plt.show()
    
    os.makedirs(save_fig_folder, exist_ok=True)
    plt.savefig(f"{save_fig_folder}/grid_" + str(key_ind) + ".png")
    

def plot_cliff_over_velocity_single_grid_time(key_ind, data, mean, save_fig_folder):
    data = np.array(data)
    
    time = data[:,0]
    motion_angle = data[:,1]

    mean_time = mean[:, 0]
    mean_motion_angle = mean[:, 1]
    
    # Plot the quiver
    plt.clf()
    plt.close('all')
    plt.figure(figsize=(10, 8))
    
    plt.scatter(time, motion_angle, color="b", alpha=0.5)
    plt.scatter(mean_time, mean_motion_angle, color="black")

    plt.title("Grid " + str(key_ind))
    # plt.show()
    
    os.makedirs(save_fig_folder, exist_ok=True)
    plt.savefig(f"{save_fig_folder}/grid_" + str(key_ind) + "_time.png")


def plot_ms_results(key_ind, data, cluster_labels, save_fig_folder):
    data = np.array(data)
    
    x = data[:,0] * np.cos(data[:,1])
    y = data[:,0] * np.sin(data[:,1])
    
    n_clusters = len(set(cluster_labels))

    plt.clf()
    plt.close('all')
    plt.figure(figsize=(10, 8))
    
    ##### For using the hsv plot map #####
    # colors = plt.cm.get_cmap('hsv', n_clusters)
    # for i in range(n_clusters):
    #     cluster_indices = np.where(cluster_labels == i)
    #     plt.quiver(np.zeros_like(x[cluster_indices]), np.zeros_like(y[cluster_indices]), 
    #             x[cluster_indices], y[cluster_indices], 
    #             color=colors(i), scale=5)
    ######################################
    
    print(cluster_labels)
    
    ##### For using the self defined color map #####
    custom_colors = ['blue', 'red', 'orange', 'green', 'purple', 'cyan', 'magenta', 'yellow']
    for label in set(cluster_labels):
        # if label == -1:  # Skip plotting for label -1
        #     continue
        cluster_indices = np.where(cluster_labels == label)
        plt.quiver(np.zeros_like(x[cluster_indices]), np.zeros_like(y[cluster_indices]), 
                x[cluster_indices], y[cluster_indices], 
                color=custom_colors[label % len(custom_colors)], scale=5)
    ######################################
    
    plt.title("Grid " + str(key_ind))
    # plt.show()
    plt.savefig(f"{save_fig_folder}/grid_" + str(key_ind) + "_ms.png")


def plot_ms_results_time(key_ind, data, cluster_labels, save_fig_folder):
    data = np.array(data)
    
    time = data[:,0]
    motion_angle = data[:,1]
    
    n_clusters = len(set(cluster_labels))

    plt.clf()
    plt.close('all')
    plt.figure(figsize=(10, 8))
    
    ##### For using the self defined color map #####
    custom_colors = ['blue', 'red', 'orange', 'green', 'purple', 'cyan', 'magenta', 'yellow']
    for label in set(cluster_labels):
        cluster_indices = np.where(cluster_labels == label)
        plt.scatter(time[cluster_indices], motion_angle[cluster_indices],
                color=custom_colors[label % len(custom_colors)])
    ######################################
    
    plt.title("Grid " + str(key_ind))
    # plt.show()
    plt.savefig(f"{save_fig_folder}/grid_" + str(key_ind) + "_ms_time.png")


def save_cliff_csv(csv_file, row) -> None:
    with open(csv_file, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(row)
        
def save_cliff_csv_rows(csv_file, rows) -> None:
    with open(csv_file, 'a', newline='') as csvfile:
        for row in rows:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(row)


def read_config_file(config_file) -> dict:
    with open(config_file, 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    return config_data

def read_cliff_map_data(cliff_map_file):
    data = pd.read_csv(cliff_map_file, header=None)
    data.columns = ["x", "y", "velocity", "motion_angle",
                    "cov1", "cov2", "cov3", "cov4", "weight",
                    "motion_ratio"]

    return data.to_numpy()