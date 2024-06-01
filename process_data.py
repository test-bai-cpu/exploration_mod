import pandas as pd
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

import online_utils as utils

class DataProcess:
    
    def __init__(
        self,
        raw_data_file: str,
        radius: float,
        step: float,
        dataset_type: str,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
    ) -> None:
        self.raw_data_file = raw_data_file
        self.radius = radius
        self.step = step
        
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        
        self.data = None
        self.grid_data = {}
        
        self._preprocess_data(dataset_type)
        self._split_data_to_grid()
        self._compute_motion_ratio()
    
    def _preprocess_data(self, dataset_type='ATC'):
        
        if dataset_type == 'ATC':
            ################### ATC dataset ###################
            data = pd.read_csv(self.raw_data_file, header=None)
            data.columns = ['time', 'person_id', 'x', 'y', 'z', 'velocity', 'motion_angle', 'facing_angle']
            ### In ATC: velocity unit is mm/s, motion_angle unit is degrees (-pi, pi), update to (0, 2pi)
            data['motion_angle'] = np.mod(data['motion_angle'], 2 * np.pi)
            ### set x, y, z, velocity columns from milimeter to meter
            data = utils.millimeter_to_meter(data, ['x', 'y', 'z', 'velocity'])
            ### We only need the x, y, velocity, and motion_angle columns
            self.data = data[['x', 'y', 'velocity', 'motion_angle']]
            ###################################################
        
        elif dataset_type == 'ATC-1HZ':
            ################## already processed 1s ATC dataset ###################
            data = pd.read_csv(self.raw_data_file, header=None)
            data.columns = ['time', 'person_id', 'x', 'y', 'velocity', 'motion_angle']
            ### In ATC: velocity unit is mm/s, motion_angle unit is degrees (-pi, pi), update to (0, 2pi)
            data['motion_angle'] = np.mod(data['motion_angle'], 2 * np.pi)
            ### set x, y, z, velocity columns from milimeter to meter
            # data = utils.millimeter_to_meter(data, ['x', 'y', 'z', 'velocity'])
            ### We only need the x, y, velocity, and motion_angle columns
            self.data = data[['x', 'y', 'velocity', 'motion_angle']]
            ##################################################

        elif dataset_type == 'MAGNI':
            ################### MAGNI dataset ###################
            data = pd.read_csv(self.raw_data_file)
            data = data.rename(columns={
                'Time': 'time',
                'ag_id': 'person_id', 
                'speed': 'velocity', 
                'theta_delta': 'motion_angle'})
            data['motion_angle'] = np.mod(data['motion_angle'], 2 * np.pi)
            self.data = data[['x', 'y', 'velocity', 'motion_angle']]
            ###################################################

        elif dataset_type == 'EDIN':
            data = pd.read_csv(self.raw_data_file, header=None)
            data.columns = ['time', 'person_id', 'x', 'y', 'velocity', 'motion_angle']
            data['motion_angle'] = np.mod(data['motion_angle'], 2 * np.pi)
            self.data = data[['x', 'y', 'velocity', 'motion_angle']]

        elif dataset_type == 'EDIN_in_folder':
            train_edin_dates = ["0106", "0107", "0111", "0113", "0114"]
            train_dfs = []
            for date in train_edin_dates:
                data = pd.read_csv(f"{self.raw_data_file}/{date}.csv", header=None)
                train_dfs.append(data)
                
            combined_df = pd.concat(train_dfs, axis=0, ignore_index=True)
            combined_df.columns = ['time', 'person_id', 'x', 'y', 'velocity', 'motion_angle']
            combined_df['motion_angle'] = np.mod(combined_df['motion_angle'], 2 * np.pi)
            self.data = combined_df[['x', 'y', 'velocity', 'motion_angle']]
        
        ##### TODO!! open question: In cliffmap matlab, speed=0 measurement will be removed
        
        if self.x_min is None or self.x_max is None or self.y_min is None or self.y_max is None:
            self._set_min_max_values()
        
    def _set_min_max_values(self):
        x_min, x_max = self.data['x'].min(), self.data['x'].max()
        y_min, y_max = self.data['y'].min(), self.data['y'].max()
        
        # self.x_min = x_min - self.step
        # self.x_max = x_max + self.step
        # self.y_min = y_min - self.step
        # self.y_max = y_max + self.step
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
    
    def _split_data_to_grid(self):
        self.grid_data = {}

        x_centers = np.arange(self.x_min + self.step / 2, self.x_max, self.step)
        y_centers = np.arange(self.y_min + self.step / 2, self.y_max, self.step)

        grid_centers = [(round(x, 3), round(y, 3)) for x in x_centers for y in y_centers]

        for grid_center in grid_centers:
            distances = np.sqrt((self.data['x'] - grid_center[0]) ** 2 + (self.data['y'] - grid_center[1]) ** 2)
            within_radius = self.data[distances <= self.radius]
            within_radius = within_radius[['velocity', 'motion_angle']]
            if not within_radius.empty:
                self.grid_data[grid_center] = GridData(grid_center, within_radius)
                
    def _compute_motion_ratio(self):
        max_motion = 0
        for grid_data in self.grid_data.values():
            data_len = len(grid_data.data)
            if data_len > max_motion:
                max_motion = data_len
        
        for grid_data in self.grid_data.values():
            grid_data.motion_ratio = len(grid_data.data) / max_motion
        
    def get_data_for_grid(self, grid_x, grid_y):
        grid_key = (grid_x, grid_y)
        
        grid_data = self.grid_data.get(grid_key, None)
        
        if grid_data is None:
            print("In procees_data.py, get_data_for_grid: grid_key not found")
        
        return grid_data
        
    def plot_data_by_color_all_grid(self):
        # Create a color map to color each grid's data differently
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.grid_data)))
        # colors = plt.cm.rainbow(np.linspace(0, 1, 15))
        # colors = [
        #     "blue", "green", "red", "cyan", "magenta", 
        #     "yellow", "black", "white", "orange", "purple", 
        #     "brown", "pink", "lime", "navy", "gray", 
        #     "olive", "maroon", "teal", "aqua", "silver", 
        #     "gold", "lavender", "beige", "ivory", "chocolate", 
        #     "salmon"]
        
        grid_centers = list(self.grid_data.keys())
        # print(len(grid_centers))
        
        for i in range(26):
            grid_center = grid_centers[i]
            color = colors[i]
            self.plot_for_one_grid(grid_center[0], grid_center[1], color=color)
        
    def plot_for_one_grid(self, grid_x, grid_y, color='blue'):
        grid_data = self.get_data_for_grid(grid_x, grid_y)
        plt.scatter(grid_data.data['x'], grid_data.data['y'], color=color, label=f"Grid {grid_data.grid_center}", alpha=1)


class GridData:
    def __init__(
        self,
        grid_center: Tuple[float, float],
        data: pd.DataFrame,
    ) -> None:
        self.grid_center = grid_center
        self.data = data
        
        self.motion_ratio = None
