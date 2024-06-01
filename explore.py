import os
import copy
import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import explore_utils as utils
from explore_utils import PixWorldConverter


matplotlib.use('TkAgg')


class Pose:
    def __init__(self, x: float = 0., y: float = 0., theta: float = 0.) -> None:
        self.x = x
        self.y = y
        self.theta = theta

    def update(self, x: float = 0., y: float = 0., theta: float = 0.) -> None:
        self.x = x
        self.y = y
        self.theta = theta


class ViewArea:
    def __init__(self, type="cone", angle: float = 120., radius: float = 200.) -> None:
        self.type = type
        self.angle = np.radians(angle)
        self.radius = radius


class ExploreMap:
    def __init__(self, config_file: str, ) -> None:
        self.config_params = utils.read_config_file(config_file)
        self.map_file = self.config_params['map_file']

        # Initial pose of the robot
        self.pose = Pose(self.config_params['x_init'],
                         self.config_params['y_init'],
                         self.config_params['theta_init'])

        self.view = ViewArea(self.config_params['view_type'],
                             self.config_params['view_angle'],
                             self.config_params['view_radius'])

        # Get image array and flip it
        img = cv2.imread(self.map_file, cv2.IMREAD_GRAYSCALE)
        self.img_array = cv2.flip(img, 0)[self.config_params['y_min']:self.config_params['y_max'],
                                          self.config_params['x_min']:self.config_params['x_max']]
        self.img_array[self.img_array < 200] = 0
        self.img_array[self.img_array >= 200] = 128
        # img = np.array(Image.open(self.map_file))
        # self.img_array = np.flipud(img[:, :, :3])

    def get_fov_cone(self, num_points: int = 100):
        angle_start = self.pose.theta - self.view.angle / 2
        angle_end = self.pose.theta + self.view.angle / 2
        pt1 = (int(self.pose.x + self.view.radius * np.cos(angle_start)),
               int(self.pose.y + self.view.radius * np.sin(angle_start)))
        pt2 = (int(self.pose.x + self.view.radius * np.cos(angle_end)),
               int(self.pose.y + self.view.radius * np.sin(angle_end)))
        angles = np.linspace(angle_start, angle_end, num_points)
        arc_points = [(int(self.pose.x + self.view.radius * np.cos(angle)),
                       int(self.pose.y + self.view.radius * np.sin(angle))) for angle in angles]
        # fov_points = np.array([(self.pose.x, self.pose.y)] +
        #                   arc_points + [(self.pose.x, self.pose.y)], np.int32)
        return pt1, pt2, arc_points

    def plot_map(self):
        visuals_info = {"resolution_pm": 0.01, "offset": [-860, -1050]}
        pix2world_conv = PixWorldConverter(visuals_info)
        # cliff_xy_pixels = pix2world_conv.convert2pixels(cliff_map_data[:, 0:2])

        # Plot FOV
        if self.view.type == "cone":
            pt1, pt2, arc_points = self.get_fov_cone()

            # Fill the FOV
            fov_points = np.array(
                [(self.pose.x, self.pose.y)] + arc_points + [(self.pose.x, self.pose.y)], np.int32)
            fov_mask = np.zeros((self.img_array.shape[0], self.img_array.shape[1]), dtype=np.uint8)
            cv2.fillPoly(fov_mask, [fov_points], 255)
            obstacle_mask = cv2.inRange(self.img_array, 0, 0)
            free_space_mask = cv2.bitwise_not(obstacle_mask)
            fov = cv2.bitwise_and(fov_mask, free_space_mask)
            self.img_array[fov > 0] = 255
            plt.imshow(self.img_array, cmap='gray', vmin=0, vmax=255)

            # Plot the FOV boundary as two lines in pixel coordinates
            plt.plot(self.pose.x, self.pose.y, 'bo', markersize=7)
            plt.plot([self.pose.x, pt1[0]], [self.pose.y, pt1[1]], 'r--', lw=3)
            plt.plot([self.pose.x, pt2[0]], [self.pose.y, pt2[1]], 'r--', lw=3)

            # Plot the arc of the FOV in pixel coordinates
            plt.plot([p[0] for p in arc_points],
                     [p[1] for p in arc_points], 'r--', lw=3)

        plt.xlim([0, 1790])
        plt.ylim([0, 767])

        # plt.show()
        plt.savefig('figures/map_check.png', bbox_inches='tight')

    def is_in_free_space(self, position, plot=False):
        pixel_color = self.img_array[position[1], position[0]]
        if_free = np.all(pixel_color > 100)
        if plot:
            self.plot_map()

        return if_free

    def get_view_point_list():
        # return a list of view points
        pass


if __name__ == "__main__":
    explore = ExploreMap('explore_config.yaml')
    explore.plot_map()

    # Check if a position is in free space, can uncomment plot line in get_if_in_free_space to visualize the position
    # explored_position = [1500, 1200, 0]
    explored_position = (400, 500, 0)
    print(explore.is_in_free_space(explored_position, plot=True))
