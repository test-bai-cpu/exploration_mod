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
from typing import Union
import torch

import pandas as pd


class PixWorldConverter:
    """Pixel to world converter"""

    def __init__(self, info: dict) -> None:
        self.resolution = info["resolution_pm"]  # 1pix -> m
        self.offset = np.array(info["offset"])

    def convert2pixels(
        self, world_locations: Union[np.array, torch.Tensor]
    ) -> Union[np.array, torch.Tensor]:

        if world_locations.ndim == 2:
            return (world_locations / self.resolution) - self.offset
        new_world_locations = [
            self.convert2pixels(world_location) for world_location in world_locations
        ]
        return (
            torch.stack(new_world_locations)
            if isinstance(world_locations, torch.Tensor)
            else np.stack(new_world_locations)
        )

    def convert2world(
        self, pix_locations: Union[np.array, torch.Tensor]
    ) -> Union[np.array, torch.Tensor]:
        return (pix_locations + self.offset) * self.resolution


def read_config_file(config_file) -> dict:
    with open(config_file, 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    return config_data


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def calc_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2-y1)**2)
