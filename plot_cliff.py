import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import online_utils as utils
import warnings
import os

from PIL import Image
from explore_utils import PixWorldConverter
from matplotlib.ticker import FixedLocator, FixedFormatter

from pprint import pprint



def read_cliff_map_data(cliff_map_file):
    try:
        data = pd.read_csv(cliff_map_file, header=None)
    except pd.errors.EmptyDataError:
        print(f"Empty data in {cliff_map_file}")
        return np.array([])
    
    data.columns = ["x", "y", "motion_angle", "velocity",
                    "cov1", "cov2", "cov3", "cov4", "weight", "motion_ratio"]
    
    return data.to_numpy()


def plot_cliff_map_with_weight(cliff_map_data):

    ## Only leave the SWND with largest weight
    max_index_list = []
    
    location = cliff_map_data[0, :2]
    weight = cliff_map_data[0, 8]
    max_weight_index = 0

    for i in range(1, len(cliff_map_data)):
        tmp_location = cliff_map_data[i, :2]
        if (tmp_location == location).all():
            tmp_weight = cliff_map_data[i, 8]
            if tmp_weight > weight:
                max_weight_index = i
                weight = tmp_weight
        else:
            max_index_list.append(max_weight_index)
            location = cliff_map_data[i, :2]
            weight = cliff_map_data[i, 8]
            max_weight_index = i

    max_index_list.append(max_weight_index)

    (u, v) = utils.pol2cart(cliff_map_data[:, 2] * 100, cliff_map_data[:, 3])
    weight = cliff_map_data[:, 8]

    colors = cliff_map_data[:, 3]  * 180 / np.pi
    colors = np.append(colors, [0, 360])
    norm = Normalize()
    norm.autoscale(colors)
    colormap = cm.hsv

    for i in range(len(cliff_map_data)):
    # for i in range(200):
        ## For only plot max weight:
        # if i in max_index_list:
            # plt.quiver(cliff_map_data[i, 0], cliff_map_data[i, 1], u[i], v[i], color=colormap(norm(colors))[i], alpha=1, cmap="hsv",angles='xy', scale_units='xy', scale=0.7)
        ## For only plot one point:
        # if cliff_map_data[i, 0] == 20 and cliff_map_data[i, 1] == -13:
        
        if weight[i] > 1:
            weight[i] = 1
        plt.quiver(cliff_map_data[i, 0], cliff_map_data[i, 1], u[i], v[i], color=colormap(norm(colors))[i], alpha=weight[i], cmap="hsv",angles='xy', scale_units='xy', scale=1, width=0.004)
        # plt.quiver(cliff_map_data[i, 0], cliff_map_data[i, 1], u[i], v[i], color=colormap(norm(colors))[i], alpha=weight[i], cmap="hsv",angles='xy', scale_units='xy', scale=0.7)


    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    cbar = plt.colorbar(sm, shrink = 0.5, ticks=[0, 90, 180, 270, 360], fraction=0.05)
    cbar.ax.tick_params(labelsize=10)
    # plt.text(100, -17,"Orientation [deg]", rotation='vertical')
    # plt.text(75, -14,"Orientation [deg]", rotation='vertical')


if __name__ == "__main__":
    observe_period = 200
    save_folder = "online_mod_res_new_figs_learning_rate_change_v2"
    
    for exp_type in ["A", "B"]:
        for observe_start_time in range(0, 10, 1):
            observe_start_time = observe_start_time * observe_period
            print("-------------------- In time period -------------------")
            print(observe_start_time, observe_start_time + observe_period)
            for cliff_type in ["all", "online", "interval"]:
                save_fig_dir = f"{save_folder}/{cliff_type}"
                os.makedirs(save_fig_dir, exist_ok=True)
                cliff_file_name = f"online_mod_res_all_dates_learning_rate_change_v2/{exp_type}_{observe_start_time}_{observe_start_time + observe_period}_{cliff_type}.csv"    
                cliff_map_data = read_cliff_map_data(cliff_file_name)
                
                if len(cliff_map_data) == 0:
                    continue

                plt.clf()
                plt.close('all')
                
                img = np.array(Image.open('thor_magni_maps/1205_SC1A_map.png'))
                spatial_layout = np.flipud(img[:, :, :3])
                plt.imshow(spatial_layout)
                
                visuals_info = {"resolution_pm" : 0.01, "offset": [-860, -1050]}
                pix2world_conv = PixWorldConverter(visuals_info)
                cliff_xy_pixels = pix2world_conv.convert2pixels(cliff_map_data[:, 0:2])
                cliff_map_data[:, 0:2] = cliff_xy_pixels
                plot_cliff_map_with_weight(cliff_map_data)
                
                plt.xlim([0, 1790])
                plt.ylim([650, 1417])
                
                plt.title(f"Cliff map at time {observe_start_time} to {observe_start_time + observe_period} with cliff type {cliff_type}")
                plt.savefig(f"{save_fig_dir}/{exp_type}_{observe_start_time}_{observe_start_time + observe_period}_{cliff_type}.png")
            
        