import pandas as pd
import numpy as np

import os, sys

from online_update_all import OnlineUpdateMoD
from observe import InThorMagniDataset


def test_online_on_atc():
    
    onlineUpdateMoD = OnlineUpdateMoD(
        config_file="config_atc_v2.yaml",
        current_cliff=None,
        output_cliff_folder=f"online_mod_res_atc_split_random_all_single_file_simple",
        save_fig_folder=f"save_fig_online")

    hour = int(sys.argv[1])
    thor_magni = InThorMagniDataset('config_atc.yaml', raw_data_file=f'atc-1s-ds-1024-split-hour/combine_for_train_all/atc_combine_{hour}.csv')
    observed_traj = thor_magni.get_observed_traj_all_area_all_time()
    onlineUpdateMoD.updateMoD(observed_traj, f"ATC1024_{hour}_{hour + 1}")


test_online_on_atc()