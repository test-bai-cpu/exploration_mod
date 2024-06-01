import os

from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import explore_utils as utils
from explore_utils import PixWorldConverter

class ExploreMap:
    
    def __init__(
        self,
        config_file: str,
    ) -> None:
        self.config_params = utils.read_config_file(config_file)
        self.map_file = self.config_params['map_file']
        
        ### get image array and flip it
        img = np.array(Image.open(self.map_file))
        self.img_array = np.flipud(img[:, :, :3])
    

    def plot_map(self, vis_area=None, robot_position=None) -> None:
        visuals_info = {"resolution_pm" : 0.01, "offset": [-860, -1050]}
        pix2world_conv = PixWorldConverter(visuals_info)
        # cliff_xy_pixels = pix2world_conv.convert2pixels(cliff_map_data[:, 0:2])

        plt.imshow(self.img_array)
        
        if vis_area and vis_area["vis_type"] == "cone":
            robot_facing_angle = vis_area["robot_facing_angle"]
            fov_angle = vis_area["fov_angle"]
            fov_radius = vis_area["fov_radius"]
            
            fov_left_bound = robot_facing_angle - fov_angle / 2
            fov_right_bound = robot_facing_angle + fov_angle / 2
            
            # Plotting the FOV boundary as two lines in pixel coordinates
            left_x = robot_position[0] + fov_radius * np.cos(fov_left_bound)
            left_y = robot_position[1] + fov_radius * np.sin(fov_left_bound)
            
            right_x = robot_position[0] + fov_radius * np.cos(fov_right_bound)
            right_y = robot_position[1] + fov_radius * np.sin(fov_right_bound)
            
            left_pix = np.array([left_x, left_y])
            right_pix = np.array([right_x, right_y])
            print(left_pix[0], left_pix[1])
            plt.plot([robot_position[0], left_pix[0]], [robot_position[1], left_pix[1]], 'r--', lw=3)
            plt.plot([robot_position[0], right_pix[0]], [robot_position[1], right_pix[1]], 'r--', lw=3)
            
            # Plot the arc of the FOV in pixel coordinates
            theta = np.linspace(fov_left_bound, fov_right_bound, 100)
            arc_x = robot_position[0] + fov_radius * np.cos(theta)
            arc_y = robot_position[1] + fov_radius * np.sin(theta)
            arc_pix = np.column_stack((arc_x, arc_y))
            plt.plot(arc_pix[:, 0], arc_pix[:, 1], 'r--', lw=3)
            
        if robot_position:
            plt.plot(robot_position[0], robot_position[1], 'bo', markersize=7)        

                
        plt.xlim([0, 1790])
        plt.ylim([650, 1417])
        
        plt.show()
        # plt.savefig('figures/map_check.png', bbox_inches='tight')

    def get_if_in_free_space(self, position):
        pixel_color = self.img_array[position[1], position[0], :]
        if_free = np.all(pixel_color > 240)
        self.plot_map(robot_position = position)

        
        
        return if_free
    
    def get_view_point_list():
        ## return a list of view points
        pass
        # return list_of_view
        
    
explore = ExploreMap('explore_config.yaml')


## check map and plot the visual area
vis_area = {
    "vis_type": "cone",  # "cone" or "rectangle"
    "robot_facing_angle": 0,  # robot facing direction (radians), 0 means facing right along the x-axis
    "fov_angle": np.radians(120),  # Field of view angle in radians
    "fov_radius": 200,  # Field of view radius
}
# explore.plot_map(vis_area=vis_area, robot_position=[1000, 1200])


## Check if a position is in free space, can uncomment plot line in get_if_in_free_space to visualize the position
# explored_position = [1500, 1200]
explored_position = [1400, 1200]
res = explore.get_if_in_free_space(explored_position)
print(res)