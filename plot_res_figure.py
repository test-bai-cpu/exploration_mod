import matplotlib.pyplot as plt
import explore_utils as utils
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Patch

import pandas as pd
import numpy as np

import os

def read_cliff_map_data(cliff_map_file):
    try:
        data = pd.read_csv(cliff_map_file, header=None)
    except pd.errors.EmptyDataError:
        print(f"Empty data in {cliff_map_file}")
        return np.array([])
    
    data.columns = ["x", "y", "velocity", "motion_angle",
                    "cov1", "cov2", "cov3", "cov4", "weight", "motion_ratio"]
    
    return data.to_numpy()

def plot_art_exp(cliff_map_file, output_fig_name):
    plt.figure(figsize=(10, 8), dpi=100)
    
    cliff_map_data = read_cliff_map_data(cliff_map_file)

    (u, v) = utils.pol2cart(cliff_map_data[:, 2], cliff_map_data[:, 3])
    weight = cliff_map_data[:, 8]

    for i in range(len(cliff_map_data)):
        print(cliff_map_data[i, 0], cliff_map_data[i, 1], u[i], v[i])
        plt.quiver(cliff_map_data[i, 0], cliff_map_data[i, 1], u[i], v[i], color='b', alpha=weight[i], angles='xy', scale_units='xy', scale=1, width=0.03)

    plt.scatter(cliff_map_data[0, 0], cliff_map_data[0, 1], c='black', s=200)

    # norm = Normalize(vmin=0, vmax=1)
    # # cmap = cm.Blues
    # colors = [(0, 0, 1, alpha) for alpha in np.linspace(0, 1, 256)]
    # blue_cmap = LinearSegmentedColormap.from_list("BlueAlphaCmap", colors, N=256)
    # sm = plt.cm.ScalarMappable(cmap=blue_cmap, norm=norm)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, ax=plt.gca(), shrink = 0.8, fraction=0.05, pad=0.06)
    # cbar.set_label('Weight', fontsize=20)
    # cbar.ax.tick_params(labelsize=20)


    plt.axis('off')
    plt.axis('equal')
    plt.xlim(-1, 3)
    plt.ylim(-1, 3)
    
    
    plt.savefig(output_fig_name, bbox_inches='tight')
    # plt.show()

def plot_art_data(art_map_file, output_fig_name):
    plt.figure(figsize=(10, 8), dpi=100)

    data = pd.read_csv(art_map_file, header=None)
    data.columns = ["time", "person_id", "x", "y", "velocity", "motion_angle"]
    data['motion_angle'] = np.mod(data['motion_angle'], 2 * np.pi)
    data = data[['time', 'x', 'y', 'velocity', 'motion_angle']]

    (u, v) = utils.pol2cart(data[['velocity']].values, data[['motion_angle']].values)
    plt.quiver(data['x'].values, data['y'].values, u, v, color="black", alpha=0.6, angles='xy', scale_units='xy', scale=1, width=0.004)

    plt.axis('off')
    plt.axis('equal')
    plt.xlim(-1, 3)
    plt.ylim(-1, 3)
    
    plt.savefig(output_fig_name, bbox_inches='tight')
    # plt.show()


# # for type in ["all", "interval", "online"]:
# for type in ["all"]:
#     save_folder = f"online_mod_res_art_v5/figs/{type}"
#     os.makedirs(save_folder, exist_ok=True)
#     for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
#         plot_art_exp(f"online_mod_res_art_v5/b1_{angle}_{type}.csv", f"{save_folder}/b1_{angle}_{type}.png")

# angle = 270
# type = "online"
# plot_art_exp(f"online_mod_res_art/b1_{angle}_{type}.csv", f"online_mod_res_art/figs/legend2.png")
# os.makedirs("online_mod_res_art/figs_v3", exist_ok=True)

# for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
#     art_map_file = f'/home/yufei/research/online_mod/online_mod/online_batch_data/artv3/b1_{angle}.csv'
#     plot_art_data(art_map_file, f"online_mod_res_art/figs_v3/raw_data_{angle}.png")