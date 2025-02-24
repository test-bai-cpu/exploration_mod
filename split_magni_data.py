import pandas as pd
import os

def split_magni_data():
    exp_type = 'A'
    data = pd.read_csv(f'thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/{exp_type}.csv')
    save_folder = "thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/split_data"
    os.makedirs(save_folder, exist_ok=True)
    # Define the time slots
    time_slots = [(i, i + 200) for i in range(0, 2000, 200)]

    # Process each time slot
    for start_time, end_time in time_slots:
        # Filter data within the current time slot
        slot_data = data[(data['Time'] >= start_time) & (data['Time'] < end_time)]
        
        if len(slot_data) == 0:
            continue
        
        # Calculate the size of each 10% segment
        segment_size = len(slot_data) // 10
        
        for i in range(10):
            test_data = slot_data.iloc[i * segment_size: (i + 1) * segment_size]
            train_data = pd.concat([slot_data.iloc[:i * segment_size], slot_data.iloc[(i + 1) * segment_size:]])
            
            # Save the test and train datasets to files
            test_filename = f'{save_folder}/{exp_type}_test_{start_time}_{end_time}_{i + 1}.csv'
            train_filename = f'{save_folder}/{exp_type}_train_{start_time}_{end_time}_{i + 1}.csv'
            
            test_data.to_csv(test_filename, index=False)
            train_data.to_csv(train_filename, index=False)


def split_magni_data_randomly_once(exp_type):
    data = pd.read_csv(f'thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/{exp_type}.csv')
    save_folder = "thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/split_data_random"
    os.makedirs(save_folder, exist_ok=True)
    # Define the time slots
    time_slots = [(i, i + 200) for i in range(0, 2000, 200)]

    # Process each time slot
    for start_time, end_time in time_slots:
        # Filter data within the current time slot
        slot_data = data[(data['Time'] >= start_time) & (data['Time'] < end_time)]
        
        if len(slot_data) == 0:
            continue
        
        test_data = slot_data.sample(frac=0.1, random_state=42)
        train_data = slot_data.drop(test_data.index)
        
        # Save the test and train datasets to files
        test_filename = f'{save_folder}/{exp_type}_test_{start_time}_{end_time}.csv'
        train_filename = f'{save_folder}/{exp_type}_train_{start_time}_{end_time}.csv'
        
        test_data.to_csv(test_filename, index=False)
        train_data.to_csv(train_filename, index=False)
        
# split_magni_data_randomly_once('A')
# split_magni_data_randomly_once('B')

def split_magni_data_by_person_id_randomly_once(exp_type):
    data = pd.read_csv(f'thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/{exp_type}.csv')
    save_folder = "thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/split_data_random_newdata"
    os.makedirs(save_folder, exist_ok=True)
    # Define the time slots
    time_slots = [(i, i + 200) for i in range(0, 2000, 200)]

    # Process each time slot
    for start_time, end_time in time_slots:
        # Filter data within the current time slot
        slot_data = data[(data['Time'] >= start_time) & (data['Time'] < end_time)]
        
        if len(slot_data) == 0:
            continue

        unique_person_ids = slot_data['person_id'].unique()
        test_person_ids = pd.Series(unique_person_ids).sample(frac=0.1, random_state=42).tolist()

        # Split the data based on the sampled person_ids
        test_data = slot_data[slot_data['person_id'].isin(test_person_ids)]
        train_data = slot_data[~slot_data['person_id'].isin(test_person_ids)]
        
        # Save the test and train datasets to files
        test_filename = f'{save_folder}/{exp_type}_test_{start_time}_{end_time}.csv'
        train_filename = f'{save_folder}/{exp_type}_train_{start_time}_{end_time}.csv'
        
        test_data.to_csv(test_filename, index=False)
        train_data.to_csv(train_filename, index=False)
        
##########################



split_magni_data_by_person_id_randomly_once("A")
split_magni_data_by_person_id_randomly_once("B")


def combine_data(hours, exp_type, special_case=False):
    data_frames = []
    
    if special_case:
        file_path_list = [
            f'thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/split_data_random/A_train_1800_2000.csv',
            f'thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/split_data_random/B_train_0_200.csv'
        ]
        for file_path in file_path_list:
            df = pd.read_csv(file_path)
            data_frames.append(df)
    else:
        for hour in hours:
            file_path = f'thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/split_data_random/{exp_type}_train_{hour}_{hour+200}.csv'
            df = pd.read_csv(file_path)
            data_frames.append(df)
    
    combined_data = pd.concat(data_frames, ignore_index=True)
    return combined_data


def combine_days_atc_for_train_all_A():
    hour_groups = {hour: list(range(0, hour + 200, 200)) for hour in range(0, 2000, 200)}
    
    print(hour_groups)
    
    # Process each group, combine the data, and save to new CSV files
    for hour, hours in hour_groups.items():
        data_frames = []
        for loop_h in hours:
            file_path = f'thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/split_data_random/A_train_{loop_h}_{loop_h+200}.csv'
            df = pd.read_csv(file_path)
            data_frames.append(df)
        
        combined_data = pd.concat(data_frames, ignore_index=True)
        combined_data.to_csv(f'thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/combine_for_train_all/A_magni_combine_{hour}_{hour+200}.csv', index=False)


# Which add all A data in B data
def combine_days_atc_for_train_all_B():
    hour_groups = {hour: list(range(0, hour + 200, 200)) for hour in range(0, 2000, 200)}
    
    print(hour_groups)
    # Process each group, combine the data, and save to new CSV files
    for hour, hours in hour_groups.items():
        data_frames = []
        file_path = f'thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/combine_for_train_all/A_magni_combine_1800_2000.csv'
        df = pd.read_csv(file_path)
        data_frames.append(df)
        for loop_h in hours:
            file_path = f'thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/split_data_random/B_train_{loop_h}_{loop_h+200}.csv'
            df = pd.read_csv(file_path)
            data_frames.append(df)
        
        combined_data = pd.concat(data_frames, ignore_index=True)
        combined_data.to_csv(f'thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/combine_for_train_all/B_magni_combine_{hour}_{hour+200}.csv', index=False)

# os.makedirs('thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/combine_for_train_all', exist_ok=True)
# combine_days_atc_for_train_all_A()
# combine_days_atc_for_train_all_B()


def combine_days_magni_for_window_average():
    hour_groups = {hour: list(range(hour - 200, hour + 200, 200)) for hour in range(0, 2000, 200)}
    hour_groups[0] = [0, 200]
    print(hour_groups)

    for exp_type in ['A', 'B']:
        # Process each group, combine the data, 10 for 9 and 10, 11 for 9,10,11, and so on, and save to new CSV files
        for hour, hours in hour_groups.items():
            special_case = False
            if exp_type == "B" and hour == 0:
                special_case = True
            combined_df = combine_data(hours, exp_type, special_case=special_case)
            combined_df.to_csv(f'thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/window_average/{exp_type}_train_{hour}_{hour+200}.csv', index=False)


# os.makedirs('thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/window_average', exist_ok=True)
# combine_days_magni_for_window_average()