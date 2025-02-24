import pandas as pd
import numpy as np

import os, sys

from online_update_all import OnlineUpdateMoD
from observe import InThorMagniDataset


def test_online():
    config_file= "config_mapf.yaml"
    onlineUpdateMoD = OnlineUpdateMoD(
        config_file=config_file,
        current_cliff=None,
        output_cliff_folder=f"online_mod_res_mapf/all",
        save_fig_folder=f"save_fig_online")
    
    for version in ["initial_split", "update_split"]:
        for batch in range(1, 11):
            thor_magni = InThorMagniDataset(config_file, raw_data_file=f'mapf_corl_rebuttal/{version}/b{batch}.csv')
            observed_traj = thor_magni.get_observed_traj_all_area_all_time()
            onlineUpdateMoD.updateMoD(observed_traj, f"{version}_b{batch}")

test_online()