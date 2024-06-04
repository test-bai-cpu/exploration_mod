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

        self.view = ViewArea(self.config_params['fov_type'],
                             self.config_params['fov_angle'],
                             self.config_params['fov_radius'])

        # Get image array and flip it
        img = cv2.imread(self.map_file, cv2.IMREAD_GRAYSCALE)
        self.img_array = cv2.flip(img, 0)[self.config_params['y_min']:self.config_params['y_max'],
                                          self.config_params['x_min']:self.config_params['x_max']]
        self.img_array[self.img_array < 200] = self.config_params['OCCUPIED']
        self.img_array[self.img_array >= 200] = self.config_params['UNKNOWN']
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
        return pt1, pt2, arc_points

    def plot_map(self, name: str = None):
        visuals_info = {"resolution_pm": 0.01, "offset": [-860, -1050]}
        pix2world_conv = PixWorldConverter(visuals_info)
        # cliff_xy_pixels = pix2world_conv.convert2pixels(cliff_map_data[:, 0:2])
        fig, ax = plt.subplots(1, 1)

        # Plot FOV
        if self.view.type == "cone":
            pt1, pt2, arc_points = self.get_fov_cone()

            # Fill the FOV
            fov_points = np.array(
                [(self.pose.x, self.pose.y)] + arc_points + [(self.pose.x, self.pose.y)], np.int32)
            fov_mask = np.zeros(
                (self.img_array.shape[0], self.img_array.shape[1]), dtype=np.uint8)
            cv2.fillPoly(fov_mask, [fov_points], self.config_params['FREE'])
            obstacle_mask = cv2.inRange(self.img_array,
                                        self.config_params['OCCUPIED'],
                                        self.config_params['OCCUPIED'])
            free_space_mask = cv2.bitwise_not(obstacle_mask)
            fov = cv2.bitwise_and(fov_mask, free_space_mask)
            self.img_array[fov > 0] = 255
            ax.imshow(self.img_array, cmap='gray',
                      vmin=self.config_params['OCCUPIED'],
                      vmax=self.config_params['FREE'])

            # Plot the FOV boundary as two lines in pixel coordinates
            ax.plot(self.pose.x, self.pose.y, 'bo', markersize=7)
            ax.plot([self.pose.x, pt1[0]], [self.pose.y, pt1[1]], 'r--', lw=3)
            ax.plot([self.pose.x, pt2[0]], [self.pose.y, pt2[1]], 'r--', lw=3)

            # Plot the arc of the FOV in pixel coordinates
            ax.plot([p[0] for p in arc_points],
                    [p[1] for p in arc_points], 'r--', lw=3)

        plt.xlim([0, 1790])
        plt.ylim([0, 767])

        # plt.show()
        name = 'map_check.png' if name is None else name
        fig.savefig('figures/{}'.format(name), bbox_inches='tight')
        plt.close()

    def is_in_free_space(self, position, plot=False):
        pixel_color = self.img_array[position[1], position[0]]
        is_free = np.all(pixel_color >= self.config_params['UNKNOWN'])
        if plot:
            self.plot_map()
        return is_free

    def is_in_bounds(self, x, y):
        return 0 <= x < self.img_array.shape[1] and 0 <= y < self.img_array.shape[0]

    def count_free_space(self):
        free_space_mask = cv2.inRange(
            self.img_array, self.config_params['FREE'], self.config_params['FREE'])
        return cv2.countNonZero(free_space_mask)

    def count_unknown_space(self):
        unknown_space_mask = cv2.inRange(
            self.img_array, self.config_params['UNKNOWN'], self.config_params['UNKNOWN'])
        return cv2.countNonZero(unknown_space_mask)

    def count_unknown_neighbors(self, coord, neighbor_size=50):
        """Count the number of unknown neighboring pixels around a coordinate"""
        y, x = coord
        half_size = neighbor_size // 2
        neighborhood = self.img_array[max(0, y-half_size): min(self.img_array.shape[0], y+half_size+1),
                                      max(0, x-half_size): min(self.img_array.shape[1], x+half_size+1)]
        return np.sum(neighborhood == self.config_params['UNKNOWN'])

    def get_frontiers(self):
        """Return the frontier image and a list of frontier coordinates"""
        free_space_mask = cv2.inRange(
            self.img_array, self.config_params['FREE'], self.config_params['FREE'])
        frontiers = cv2.Canny(
            free_space_mask, self.config_params['OCCUPIED'], self.config_params['FREE'])
        frontier_coords = np.column_stack(np.where(frontiers > 0))
        frontier_coords = [tuple(coord) for coord in frontier_coords]
        if (self.pose.y, self.pose.x) in frontier_coords:
            frontier_coords.remove((self.pose.y, self.pose.x))
        return frontiers, frontier_coords

    def get_target_frontier_coord(self, frontier_coords, w_dist: float = 0.1):
        """Sample the target frontier based on the neighboring area of unknown space and distance to robot position"""
        scores = []
        target_frontier_coord = None

        for coord in frontier_coords:
            unknown_count = self.count_unknown_neighbors(coord)
            distance = utils.calc_distance(
                self.pose.x, self.pose.y, coord[1], coord[0])
            scores.append(unknown_count - w_dist * distance)

        # Converting scores into probability distribution with softmax
        scores = np.array(scores)
        probabilities = np.exp(scores - np.max(scores))
        probabilities /= np.sum(probabilities)

        coord_idx = np.random.choice(len(frontier_coords), p=probabilities)
        target_frontier_coord = frontier_coords[coord_idx]

        assert self.is_in_bounds(
            target_frontier_coord[1], target_frontier_coord[0]), "Frontier coordinate out of bounds!"
        return target_frontier_coord

    def get_view_point_list(self):
        """Return a list of view points"""
        view_info = {"time_step": 0, 
                     "observation_time": 20, 
                     "pose": self.pose,
                     "fov": self.view}
        view_points = [view_info]
        step = 0
        self.plot_map(name='/explore/{}.png'.format(step))

        while True:
            _, frontier_coords = self.get_frontiers()
            if not frontier_coords:
                print("Exploration complete!")
                break

            target_frontier = self.get_target_frontier_coord(
                frontier_coords, w_dist=self.config_params['w_dist'])
            # (Temporal solution) directly jump to the target frontier
            heading_new = np.arctan2(target_frontier[0] - self.pose.y,
                                     target_frontier[1] - self.pose.x)
            self.pose.update(
                x=target_frontier[1], y=target_frontier[0], theta=heading_new)
            self.plot_map(name='/explore/{}.png'.format(step))
            free_count = self.count_free_space()
            unknown_count = self.count_unknown_space()
            print("Step: {}, Target frontier: {}, Explored free space: {}, Rest unknown space: {}".format(
                step, target_frontier, free_count, unknown_count))
            view_info['time_step'] = step
            view_info['pose'] = self.pose
            view_points.append(view_info)

            # (Temporal) termination condition
            if unknown_count <= np.sum(self.img_array) * 0.000005:
                print("Rest unknown space too small, exploration terminated!")
                break
            step += 1

        return view_points


if __name__ == "__main__":
    explore = ExploreMap('explore_config.yaml')
    view_points = explore.get_view_point_list()
    print("View points: ", view_points)
