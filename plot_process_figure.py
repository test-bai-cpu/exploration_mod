import os

import pandas as pd
import numpy as np

import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from PIL import Image
from explore_utils import PixWorldConverter
from matplotlib.ticker import FixedLocator, FixedFormatter
import plot_cliff

import explore_utils as utils

def plot_fov(robot_position, robot_facing_angle, fov_angle, fov_radius, observed_traj=None, fig_name="process"):
    plt.clf()
    plt.close('all')
    
    img = np.array(Image.open('thor_magni_maps/1205_SC1A_map.png'))
    spatial_layout = np.flipud(img[:, :, :3])
    plt.imshow(spatial_layout)
    
    visuals_info = {"resolution_pm" : 0.01, "offset": [-860, -1050]}
    converter = PixWorldConverter(visuals_info)

    # Convert robot position to pixel coordinates
    robot_pos_pix = converter.convert2pixels(np.array([robot_position]))
    # print(robot_position, robot_pos_pix)

    # Plot the field of view
    fov_left_bound = robot_facing_angle - fov_angle / 2
    fov_right_bound = robot_facing_angle + fov_angle / 2
    
    # Plotting the FOV boundary as two lines in pixel coordinates
    left_x = robot_position[0] + fov_radius * np.cos(fov_left_bound)
    left_y = robot_position[1] + fov_radius * np.sin(fov_left_bound)
    
    right_x = robot_position[0] + fov_radius * np.cos(fov_right_bound)
    right_y = robot_position[1] + fov_radius * np.sin(fov_right_bound)
    
    left_pix = converter.convert2pixels(np.array([[left_x, left_y]]))
    right_pix = converter.convert2pixels(np.array([[right_x, right_y]]))
    
    plt.plot([robot_pos_pix[0][0], left_pix[0][0]], [robot_pos_pix[0][1], left_pix[0][1]], 'r--', lw=4)
    plt.plot([robot_pos_pix[0][0], right_pix[0][0]], [robot_pos_pix[0][1], right_pix[0][1]], 'r--', lw=4)
    
    
    # Plot the arc of the FOV in pixel coordinates
    theta = np.linspace(fov_left_bound, fov_right_bound, 100)
    arc_x = robot_position[0] + fov_radius * np.cos(theta)
    arc_y = robot_position[1] + fov_radius * np.sin(theta)
    arc_pix = converter.convert2pixels(np.column_stack((arc_x, arc_y)))
    plt.plot(arc_pix[:, 0], arc_pix[:, 1], 'r--', lw=4)

    # Plot the robot position in pixel coordinates
    plt.plot(robot_pos_pix[0][0], robot_pos_pix[0][1], 'bo', markersize=10)  # Robot's position
    
    
    if observed_traj is not None:
        # Convert observed trajectories to pixel coordinates
        observed_traj_pix = converter.convert2pixels(observed_traj[['x', 'y']].values)
        
        # Plot the filtered dynamics (x, y positions) in pixel coordinates
        # plt.plot(observed_traj_pix[:, 0], observed_traj_pix[:, 1], 'go')  # Observed trajectories
            

        (u, v) = utils.pol2cart(observed_traj[['velocity']].values * 100, observed_traj[['motion_angle']].values)
        color = observed_traj[['motion_angle']].values
        # print(color)
        plt.quiver(observed_traj_pix[:, 0], observed_traj_pix[:, 1], u, v, color, alpha=1, cmap="hsv", angles='xy', scale_units='xy', scale=3)


    # print(converter.convert2world(np.array([robot_pos_pix[0]])))
    # Convert the pixel tick labels to world coordinates (meters)
    ax = plt.gca()
    x_ticks_pix = ax.get_xticks()
    y_ticks_pix = ax.get_yticks()
    
    # Convert x ticks
    x_ticks_pix_pairs = np.column_stack((x_ticks_pix, np.zeros_like(x_ticks_pix)))
    x_ticks_world = converter.convert2world(x_ticks_pix_pairs)[:, 0]

    # Convert y ticks
    y_ticks_pix_pairs = np.column_stack((np.zeros_like(y_ticks_pix), y_ticks_pix))
    y_ticks_world = converter.convert2world(y_ticks_pix_pairs)[:, 1]
    
    ax.xaxis.set_major_locator(FixedLocator(x_ticks_pix))
    ax.xaxis.set_major_formatter(FixedFormatter([f'{x:.1f}' for x in x_ticks_world]))
    ax.yaxis.set_major_locator(FixedLocator(y_ticks_pix))
    ax.yaxis.set_major_formatter(FixedFormatter([f'{y:.1f}' for y in y_ticks_world]))
    
    # plt.axis('equal')
    plt.axis("off")

    plt.tight_layout()
    # plt.gca().invert_yaxis()
    

    
    if observed_traj is not None:
        plt.xlim([800, 1200])
        plt.ylim([800, 1400])
    else:
        plt.xlim([0, 1790])
        plt.ylim([650, 1417])
    
    # plt.show()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f'figures/{fig_name}.png')


def plot_fov_cliff(robot_position, robot_facing_angle, fov_angle, fov_radius, cliffmap=None, fig_name="cliff"):
    plt.clf()
    plt.close('all')
    
    img = np.array(Image.open('thor_magni_maps/1205_SC1A_map.png'))
    spatial_layout = np.flipud(img[:, :, :3])
    plt.imshow(spatial_layout)
    
    visuals_info = {"resolution_pm" : 0.01, "offset": [-860, -1050]}
    converter = PixWorldConverter(visuals_info)

    # Convert robot position to pixel coordinates
    robot_pos_pix = converter.convert2pixels(np.array([robot_position]))
    # print(robot_position, robot_pos_pix)

    # Plot the field of view
    fov_left_bound = robot_facing_angle - fov_angle / 2
    fov_right_bound = robot_facing_angle + fov_angle / 2
    
    # Plotting the FOV boundary as two lines in pixel coordinates
    left_x = robot_position[0] + fov_radius * np.cos(fov_left_bound)
    left_y = robot_position[1] + fov_radius * np.sin(fov_left_bound)
    
    right_x = robot_position[0] + fov_radius * np.cos(fov_right_bound)
    right_y = robot_position[1] + fov_radius * np.sin(fov_right_bound)
    
    left_pix = converter.convert2pixels(np.array([[left_x, left_y]]))
    right_pix = converter.convert2pixels(np.array([[right_x, right_y]]))
    
    plt.plot([robot_pos_pix[0][0], left_pix[0][0]], [robot_pos_pix[0][1], left_pix[0][1]], 'r--', lw=4)
    plt.plot([robot_pos_pix[0][0], right_pix[0][0]], [robot_pos_pix[0][1], right_pix[0][1]], 'r--', lw=4)
    
    # Plot the arc of the FOV in pixel coordinates
    theta = np.linspace(fov_left_bound, fov_right_bound, 100)
    arc_x = robot_position[0] + fov_radius * np.cos(theta)
    arc_y = robot_position[1] + fov_radius * np.sin(theta)
    arc_pix = converter.convert2pixels(np.column_stack((arc_x, arc_y)))
    plt.plot(arc_pix[:, 0], arc_pix[:, 1], 'r--', lw=4)

    # Plot the robot position in pixel coordinates
    plt.plot(robot_pos_pix[0][0], robot_pos_pix[0][1], 'bo', markersize=10)  # Robot's position
    

    try:
        cliff_map_data = pd.read_csv(cliffmap, header=None)
    except pd.errors.EmptyDataError:
        print(f"Empty data in {cliffmap}")
        return np.array([])
    
    cliff_map_data.columns = ["x", "y", "motion_angle", "velocity",
                    "cov1", "cov2", "cov3", "cov4", "weight", "motion_ratio"]
    
    cliff_map_data.loc[:, 'in_fov'] = cliff_map_data.apply(
        lambda row: in_fov(row['x'], row['y'], robot_position, robot_facing_angle, fov_angle, fov_radius), axis=1
    )
    
    cliff_map_data = cliff_map_data[cliff_map_data['in_fov']].drop(columns=['in_fov'])
    cliff_map_data = cliff_map_data.to_numpy()
    cliff_xy_pixels = converter.convert2pixels(cliff_map_data[:, 0:2])
    
    cliff_map_data[:, 0:2] = cliff_xy_pixels
    plot_cliff.plot_cliff_map_with_weight(cliff_map_data)

    # print(converter.convert2world(np.array([robot_pos_pix[0]])))
    # Convert the pixel tick labels to world coordinates (meters)
    ax = plt.gca()
    x_ticks_pix = ax.get_xticks()
    y_ticks_pix = ax.get_yticks()
    
    # Convert x ticks
    x_ticks_pix_pairs = np.column_stack((x_ticks_pix, np.zeros_like(x_ticks_pix)))
    x_ticks_world = converter.convert2world(x_ticks_pix_pairs)[:, 0]

    # Convert y ticks
    y_ticks_pix_pairs = np.column_stack((np.zeros_like(y_ticks_pix), y_ticks_pix))
    y_ticks_world = converter.convert2world(y_ticks_pix_pairs)[:, 1]
    
    ax.xaxis.set_major_locator(FixedLocator(x_ticks_pix))
    ax.xaxis.set_major_formatter(FixedFormatter([f'{x:.1f}' for x in x_ticks_world]))
    ax.yaxis.set_major_locator(FixedLocator(y_ticks_pix))
    ax.yaxis.set_major_formatter(FixedFormatter([f'{y:.1f}' for y in y_ticks_world]))
    
    # plt.axis('equal')
    plt.axis("off")

    plt.tight_layout()
    # plt.gca().invert_yaxis()
    
    # plt.xlim([0, 1790])
    # plt.ylim([650, 1417])
    
    plt.xlim([800, 1250])
    plt.ylim([800, 1400])
    
    # plt.xlim([0, 1850])
    # plt.ylim([650, 1417])
    
    # plt.show()
    os.makedirs("figures/cliffmap", exist_ok=True)
    plt.savefig(f'figures/cliffmap/{fig_name}.png')


def plot_full_cliff(cliffmap, fig_dir, map_dir):
    plt.clf()
    plt.close('all')
    
    img = np.array(Image.open(map_dir))
    spatial_layout = np.flipud(img[:, :, :3])
    plt.imshow(spatial_layout)
    
    visuals_info = {"resolution_pm" : 0.01, "offset": [-860, -1050]}
    converter = PixWorldConverter(visuals_info)

    try:
        cliff_map_data = pd.read_csv(cliffmap, header=None)
    except pd.errors.EmptyDataError:
        print(f"Empty data in {cliffmap}")
        return np.array([])
    
    cliff_map_data.columns = ["x", "y", "motion_angle", "velocity",
                    "cov1", "cov2", "cov3", "cov4", "weight", "motion_ratio"]
    
    cliff_map_data = cliff_map_data.to_numpy()
    cliff_xy_pixels = converter.convert2pixels(cliff_map_data[:, 0:2])
    
    cliff_map_data[:, 0:2] = cliff_xy_pixels
    plot_cliff.plot_cliff_map_with_weight(cliff_map_data)

    # print(converter.convert2world(np.array([robot_pos_pix[0]])))
    # Convert the pixel tick labels to world coordinates (meters)
    ax = plt.gca()
    x_ticks_pix = ax.get_xticks()
    y_ticks_pix = ax.get_yticks()
    
    # Convert x ticks
    x_ticks_pix_pairs = np.column_stack((x_ticks_pix, np.zeros_like(x_ticks_pix)))
    x_ticks_world = converter.convert2world(x_ticks_pix_pairs)[:, 0]

    # Convert y ticks
    y_ticks_pix_pairs = np.column_stack((np.zeros_like(y_ticks_pix), y_ticks_pix))
    y_ticks_world = converter.convert2world(y_ticks_pix_pairs)[:, 1]
    
    ax.xaxis.set_major_locator(FixedLocator(x_ticks_pix))
    ax.xaxis.set_major_formatter(FixedFormatter([f'{x:.1f}' for x in x_ticks_world]))
    ax.yaxis.set_major_locator(FixedLocator(y_ticks_pix))
    ax.yaxis.set_major_formatter(FixedFormatter([f'{y:.1f}' for y in y_ticks_world]))
    
    # plt.axis('equal')
    plt.axis("off")

    plt.tight_layout()
    # plt.gca().invert_yaxis()
    
    plt.xlim([0, 1790])
    plt.ylim([650, 1417])
    
    # plt.xlim([0, 1850])
    # plt.ylim([650, 1417])
    
    # plt.show()
    plt.savefig(fig_dir, bbox_inches='tight')
    

def in_fov(x, y, robot_pos, facing_angle, fov_angle, fov_radius):
    rel_pos = np.array([x, y]) - robot_pos
    distance = np.linalg.norm(rel_pos)
    angle = np.arctan2(rel_pos[1], rel_pos[0])
    
    angle_diff = (angle - facing_angle + np.pi) % (2 * np.pi) - np.pi
    
    return (distance <= fov_radius) and (abs(angle_diff) <= fov_angle / 2)
    

def get_observed_traj(data, robot_position, robot_facing_angle, fov_angle, fov_radius, observe_start_time, observe_period):
    # Filter data based on observation time window
    time_filtered_df = data[(data['time'] >= observe_start_time) & (data['time'] < observe_start_time + observe_period)].copy()
    
    # Filter data within the robot's field of view
    time_filtered_df.loc[:, 'in_fov'] = time_filtered_df.apply(
        lambda row: in_fov(row['x'], row['y'], robot_position, robot_facing_angle, fov_angle, fov_radius), axis=1
    )
    
    df_in_fov = time_filtered_df[time_filtered_df['in_fov']].drop(columns=['in_fov'])
    
    return df_in_fov
    

def plot_main_process_figure_magni():
    # robot_position = np.array([0, 1])  # robot view location, 2d
    robot_position = np.array([1, 2.8])  # robot view location, 2d
    robot_facing_angle = -1  # robot facing direction (radians), 0 means facing right along the x-axis
    fov_angle = np.radians(120)  # Field of view angle in radians
    fov_radius = 2  # Field of view radius
    observe_period = 200  # Observation period
    
    observe_start_time = 0
    
    # data = pd.read_csv('thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/B.csv')
    # data = data.rename(columns={
    #     'Time': 'time',
    #     'ag_id': 'person_id', 
    #     'speed': 'velocity', 
    #     'theta_delta': 'motion_angle'})
    # data['motion_angle'] = np.mod(data['motion_angle'], 2 * np.pi)
    # data = data[['time', 'x', 'y', 'velocity', 'motion_angle']]
    
    #### Plor where is the observation area ####
    # plot_fov(robot_position, robot_facing_angle, fov_angle, fov_radius)
    

    ##### Plot observation traj #####
    # # observe_start_time = 
    # observed_traj = get_observed_traj(data, robot_position, robot_facing_angle, fov_angle, fov_radius, observe_start_time, observe_period) 
    # plot_fov(robot_position, robot_facing_angle, fov_angle, fov_radius, observed_traj, fig_name="get_data")
    

    ##### Plot fov cliff map #####
    # os.makedirs("figures/cliffmap", exist_ok=True)
    # save_folder = "figures/cliffmap"
    
    # for exp_type in ["A", "B"]:
    #     for observe_start_time in range(0, 10, 1):
    #         observe_start_time = observe_start_time * observe_period
    #         plot_fov_cliff(robot_position, robot_facing_angle, fov_angle, fov_radius, cliffmap=f'online_mod_res_magni/{exp_type}_{observe_start_time}_{observe_start_time + observe_period}_online.csv', fig_name=f"cliffmap_{exp_type}_{observe_start_time}_{observe_start_time + observe_period}_online")


    ##### Plot full cliff map #####
    model_type = "online"
    save_folder = f"online_mod_res_magni_test1/figures/{model_type}"
    os.makedirs(save_folder, exist_ok=True)
    
    for exp_type in ["A", "B"]:
        map_dir = f'thor_magni_maps/1205_SC1{exp_type}_map.png'
        for observe_start_time in range(0, 10, 1):
            observe_start_time = observe_start_time * observe_period
            plot_full_cliff(
                f'online_mod_res_magni_test1/{exp_type}_{observe_start_time}_{observe_start_time + observe_period}_{model_type}.csv', 
                f"{save_folder}/cliffmap_{exp_type}_{observe_start_time}_{observe_start_time + observe_period}_{model_type}.png", 
                map_dir)


def plot_full_cliff_atc(cliffmap, fig_dir):
    plt.clf()
    plt.close('all')
    
    plt.figure(figsize=(10, 6), dpi=100)
    plt.subplot(111, facecolor='grey')
    img = plt.imread("localization_grid_white.jpg")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255, extent=[-60, 80, -40, 20])
    
    try:
        cliff_map_data = pd.read_csv(cliffmap, header=None)
    except pd.errors.EmptyDataError:
        print(f"Empty data in {cliffmap}")
        return np.array([])
    
    cliff_map_data.columns = ["x", "y", "motion_angle", "velocity",
                    "cov1", "cov2", "cov3", "cov4", "weight", "motion_ratio"]
    
    cliff_map_data = cliff_map_data.to_numpy()
    
    plot_cliff.plot_cliff_map_with_weight_ATC(cliff_map_data)
    
    plt.axis("off")
    
    plt.savefig(fig_dir, bbox_inches='tight')
    # plt.show()
    

def plot_main_process_figure_atc():
    observe_period = 200  # Observation period
    
    observe_start_time = 0

    ##### Plot full cliff map #####
    model_type = "online"
    save_folder = f"online_mod_res_atc/figures/{model_type}"
    os.makedirs(save_folder, exist_ok=True)
    
    for hour in range(9, 21, 1):
        observe_start_time = observe_start_time * observe_period
        plot_full_cliff_atc(
            f'online_mod_res_atc/ATC1024_{hour}_{hour+1}_{model_type}.csv', 
            f"{save_folder}/cliffmap_ATC1024_{hour}_{hour+1}_{model_type}.pdf")

plot_main_process_figure_magni()


# plot_main_process_figure_atc()