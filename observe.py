import os

import pandas as pd
import numpy as np

import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from PIL import Image
from explore_utils import PixWorldConverter
from matplotlib.ticker import FixedLocator, FixedFormatter

import explore_utils as utils

class InThorMagniDataset():

    def __init__(
        self,
        config_file: str,
        raw_data_file: str,
    ) -> None:
        self.config_params = utils.read_config_file(config_file)
        self.raw_data_file = raw_data_file
        self.map_file = self.config_params['map_file']
        self.raw_dataset = self.config_params['raw_dataset']
        self._load_dataset()
    
    def _load_dataset(self):
        if self.raw_dataset == 'MAGNI':
            data = pd.read_csv(self.raw_data_file)
            data = data.rename(columns={
                'Time': 'time',
                'ag_id': 'person_id', 
                'speed': 'velocity', 
                'theta_delta': 'motion_angle'})
            data['motion_angle'] = np.mod(data['motion_angle'], 2 * np.pi)
            self.data = data[['time', 'x', 'y', 'velocity', 'motion_angle']]
        
        elif self.raw_dataset == 'ATC':
            data = pd.read_csv(self.raw_data_file, header=None)
            data.columns = ["time", "person_id", "x", "y", "velocity", "motion_angle"]
            data['motion_angle'] = np.mod(data['motion_angle'], 2 * np.pi)
            self.data = data[['time', 'x', 'y', 'velocity', 'motion_angle']]
            
        elif self.raw_dataset == 'MAGNIv2':
            data = pd.read_csv(self.raw_data_file, header=None)
            data.columns = ["time", "person_id", "x", "y", "velocity", "motion_angle"]
            data['motion_angle'] = np.mod(data['motion_angle'], 2 * np.pi)
            self.data = data[['time', 'x', 'y', 'velocity', 'motion_angle']]
            
        elif self.raw_dataset == "MAPF":
            data = pd.read_csv(self.raw_data_file, header=None)
            data.columns = ["time", "person_id", "x", "y", "velocity", "motion_angle"]
            data['motion_angle'] = np.mod(data['motion_angle'], 2 * np.pi)
            self.data = data[['time', 'x', 'y', 'velocity', 'motion_angle']]
            

    def in_fov(self, x, y, robot_pos, facing_angle, fov_angle, fov_radius):
        rel_pos = np.array([x, y]) - robot_pos
        distance = np.linalg.norm(rel_pos)
        angle = np.arctan2(rel_pos[1], rel_pos[0])
        
        angle_diff = (angle - facing_angle + np.pi) % (2 * np.pi) - np.pi
        
        return (distance <= fov_radius) and (abs(angle_diff) <= fov_angle / 2)
    
    def in_region(self, x, y, obs_x, obs_y, delta_x, delta_y):
        return (x >= obs_x and x <= obs_x + delta_x and y >= obs_y and y <= obs_y + delta_y)

    def get_observed_traj(self, robot_position, robot_facing_angle, fov_angle, fov_radius, observe_start_time, observe_period):
        # Filter data based on observation time window
        time_filtered_df = self.data[(self.data['time'] >= observe_start_time) & (self.data['time'] < observe_start_time + observe_period)].copy()
        
        # Filter data within the robot's field of view
        time_filtered_df.loc[:, 'in_fov'] = time_filtered_df.apply(
            lambda row: self.in_fov(row['x'], row['y'], robot_position, robot_facing_angle, fov_angle, fov_radius), axis=1
        )
        
        df_in_fov = time_filtered_df[time_filtered_df['in_fov']].drop(columns=['in_fov'])
        
        return df_in_fov
        

    def get_observed_traj_region(self, obs_x, obs_y, delta_x, delta_y, observe_start_time, observe_period):
        # Filter data based on observation time window
        time_filtered_df = self.data[(self.data['time'] >= observe_start_time) & (self.data['time'] < observe_start_time + observe_period)].copy()
        # Filter data within the robot's field of view
        time_filtered_df.loc[:, 'in_fov'] = time_filtered_df.apply(
            lambda row: self.in_region(row['x'], row['y'], obs_x, obs_y, delta_x, delta_y), axis=1
        )
        
        df_in_fov = time_filtered_df[time_filtered_df['in_fov']].drop(columns=['in_fov'])
        
        return df_in_fov

    def get_observed_traj_region_all_time(self, obs_x, obs_y, delta_x, delta_y):
        # Filter data based on observation time window
        time_filtered_df = self.data.copy()
        
        # Filter data within the robot's field of view
        time_filtered_df.loc[:, 'in_fov'] = time_filtered_df.apply(
            lambda row: self.in_region(row['x'], row['y'], obs_x, obs_y, delta_x, delta_y), axis=1
        )
        
        df_in_fov = time_filtered_df[time_filtered_df['in_fov']].drop(columns=['in_fov'])
        
        return df_in_fov
    
    def get_observed_traj_all_area_all_time(self):
        df_in_fov = self.data.copy()
        
        return df_in_fov

    def get_observed_traj_all_area(self, obs_x, obs_y, delta_x, delta_y, observe_start_time, observe_period):
        # Filter data based on observation time window
        time_filtered_df = self.data[(self.data['time'] >= observe_start_time) & (self.data['time'] < observe_start_time + observe_period)].copy()
        
        df_in_fov = time_filtered_df
        
        return df_in_fov


    def plot_region_atc(self, obs_x, obs_y, delta_x, delta_y, observed_traj, fig_name):
        plt.clf()
        plt.close('all')
        
        plt.figure(figsize=(10, 8), dpi=200)
        plt.subplot(111)
        img = plt.imread("localization_grid_white.jpg")
        plt.imshow(img, cmap='gray', vmin=0, vmax=255, extent=[-60, 80, -40, 20])
        
        bottom_left = (obs_x, obs_y)
        bottom_right = (obs_x + delta_x, obs_y)
        top_left = (obs_x, obs_y + delta_y)
        top_right = (obs_x + delta_x, obs_y + delta_y)
        
        # Plot the four sides of the rectangle using dashed lines
        plt.plot([bottom_left[0], bottom_right[0]], [bottom_left[1], bottom_right[1]], 'r--')
        plt.plot([bottom_right[0], top_right[0]], [bottom_right[1], top_right[1]], 'r--')
        plt.plot([top_right[0], top_left[0]], [top_right[1], top_left[1]], 'r--')
        plt.plot([top_left[0], bottom_left[0]], [top_left[1], bottom_left[1]], 'r--')
    

        (u, v) = utils.pol2cart(observed_traj[['velocity']].values, observed_traj[['motion_angle']].values)
        color = observed_traj[['motion_angle']].values
        
        colors = observed_traj[['motion_angle']].values  * 180 / np.pi
        colors = np.append(colors, [0, 360])
        norm = Normalize()
        norm.autoscale(colors)
        colormap = cm.hsv
        
        plt.quiver(observed_traj['x'], observed_traj['y'], u, v, color=colormap(norm(colors)), alpha=1, cmap="hsv", angles='xy', scale_units='xy', scale=5)
        sm = cm.ScalarMappable(cmap=colormap, norm=norm)

        cbar = plt.colorbar(sm, shrink = 0.5, ticks=[0, 90, 180, 270, 360], fraction=0.05)
        cbar.ax.tick_params(labelsize=10)
        
        # plt.axis('equal')
        # plt.axis("off")
        plt.xlim([28, 33])
        plt.ylim([-20, -15])

        plt.tight_layout()
        
        # plt.text(2060, 830,"Orientation [deg]", rotation='vertical')
        
        # plt.show()
        os.makedirs("figures_ATC", exist_ok=True)
        plt.savefig(f'figures_ATC/{fig_name}.png')


    def plot_region(self, obs_x, obs_y, delta_x, delta_y, observed_traj, fig_name):
        plt.clf()
        plt.close('all')
        
        img = np.array(Image.open(self.map_file))
        spatial_layout = np.flipud(img[:, :, :3])
        plt.imshow(spatial_layout)
        
        visuals_info = {"resolution_pm" : 0.01, "offset": [-860, -1050]}
        converter = PixWorldConverter(visuals_info)

        robot_pos_pix = converter.convert2pixels(np.array([[obs_x, obs_y]]))
        # print(robot_pos_pix)
        
        obs_x = robot_pos_pix[0][0]
        obs_y = robot_pos_pix[0][1]
        delta_x = delta_x / 0.01
        delta_y = delta_y / 0.01
        
        bottom_left = (obs_x, obs_y)
        bottom_right = (obs_x + delta_x, obs_y)
        top_left = (obs_x, obs_y + delta_y)
        top_right = (obs_x + delta_x, obs_y + delta_y)
        
        # Plot the four sides of the rectangle using dashed lines
        plt.plot([bottom_left[0], bottom_right[0]], [bottom_left[1], bottom_right[1]], 'r--')
        plt.plot([bottom_right[0], top_right[0]], [bottom_right[1], top_right[1]], 'r--')
        plt.plot([top_right[0], top_left[0]], [top_right[1], top_left[1]], 'r--')
        plt.plot([top_left[0], bottom_left[0]], [top_left[1], bottom_left[1]], 'r--')
    
        
        # Convert observed trajectories to pixel coordinates
        observed_traj_pix = converter.convert2pixels(observed_traj[['x', 'y']].values)
        
        # Plot the filtered dynamics (x, y positions) in pixel coordinates
        # plt.plot(observed_traj_pix[:, 0], observed_traj_pix[:, 1], 'go')  # Observed trajectories
            

        (u, v) = utils.pol2cart(observed_traj[['velocity']].values * 100, observed_traj[['motion_angle']].values)
        color = observed_traj[['motion_angle']].values
        
        colors = observed_traj[['motion_angle']].values  * 180 / np.pi
        colors = np.append(colors, [0, 360])
        norm = Normalize()
        norm.autoscale(colors)
        colormap = cm.hsv
        
        plt.quiver(observed_traj_pix[:, 0], observed_traj_pix[:, 1], u, v, color=colormap(norm(colors)), alpha=1, cmap="hsv", angles='xy', scale_units='xy', scale=5)
        sm = cm.ScalarMappable(cmap=colormap, norm=norm)

        cbar = plt.colorbar(sm, shrink = 0.5, ticks=[0, 90, 180, 270, 360], fraction=0.05)
        cbar.ax.tick_params(labelsize=10)
        
        
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
        # plt.axis("off")

        plt.tight_layout()
        # plt.gca().invert_yaxis()
        
        plt.text(2060, 830,"Orientation [deg]", rotation='vertical')
        
        plt.xlim([0, 1790])
        plt.ylim([650, 1417])
        
        # plt.show()
        os.makedirs("figures", exist_ok=True)
        plt.savefig(f'figures/{fig_name}.png')


    def plot_fov(self, robot_position, robot_facing_angle, fov_angle, fov_radius, observed_traj, fig_name):
        plt.clf()
        plt.close('all')
        
        img = np.array(Image.open(self.map_file))
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
        
        plt.plot([robot_pos_pix[0][0], left_pix[0][0]], [robot_pos_pix[0][1], left_pix[0][1]], 'k--', lw=5)
        plt.plot([robot_pos_pix[0][0], right_pix[0][0]], [robot_pos_pix[0][1], right_pix[0][1]], 'k--', lw=5)
        
        
        # Plot the arc of the FOV in pixel coordinates
        theta = np.linspace(fov_left_bound, fov_right_bound, 100)
        arc_x = robot_position[0] + fov_radius * np.cos(theta)
        arc_y = robot_position[1] + fov_radius * np.sin(theta)
        arc_pix = converter.convert2pixels(np.column_stack((arc_x, arc_y)))
        plt.plot(arc_pix[:, 0], arc_pix[:, 1], 'k--', lw=5)

        # Plot the robot position in pixel coordinates
        plt.plot(robot_pos_pix[0][0], robot_pos_pix[0][1], 'bo', markersize=10)  # Robot's position
        
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
        # plt.axis("off")

        plt.tight_layout()
        # plt.gca().invert_yaxis()
        
        plt.xlim([0, 1790])
        plt.ylim([650, 1417])
        
        # plt.show()
        os.makedirs("figures", exist_ok=True)
        plt.savefig(f'figures/{fig_name}.png')
        

# if __name__ == '__main__':
#     thor_magni = InThorMagniDataset('config.yaml')

#     robot_position = np.array([0, 0])  # robot view location, 2d
#     robot_facing_angle = 0  # robot facing direction (radians), 0 means facing right along the x-axis
#     fov_angle = np.radians(120)  # Field of view angle in radians
#     fov_radius = 2  # Field of view radius
#     observe_period = 60  # Observation period

#     ##### For cone-shaped field of view #####
#     # observed_traj = thor_magni.get_observed_traj(robot_position, robot_facing_angle, fov_angle, fov_radius, observe_start_time, observe_period)    
#     # print(observed_traj)
#     # thor_magni.plot_fov(robot_position, robot_facing_angle, fov_angle, fov_radius, observed_traj)
    
    
#     #### For rectangular region ####
#     obs_x = -7
#     obs_y = -3.5
#     delta_x = 5
#     delta_y = 5
    
#     os.makedirs("figures", exist_ok=True)
    
#     observe_start_time = 0  # Observation start time
#     for observe_start_time in range(0, 16, 1):
#         observe_start_time = observe_start_time * 60
#         print(observe_start_time, observe_start_time + observe_period)
#         observed_traj = thor_magni.get_observed_traj_region(obs_x, obs_y, delta_x, delta_y, observe_start_time, observe_period)
#         # observed_traj = thor_magni.get_observed_traj_all_area(obs_x, obs_y, delta_x, delta_y, observe_start_time, observe_period)
        
#         # print(observed_traj)
#         thor_magni.plot_region(obs_x, obs_y, delta_x, delta_y, observed_traj, f"{observe_start_time}_{observe_start_time + observe_period}")
        