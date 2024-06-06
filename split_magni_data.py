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




def combine_data(hours, exp_type):
    data_frames = []
    for hour in hours:
        file_path = f'thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/split_data_random/{exp_type}_train_{hour}_{hour+200}.csv'
        df = pd.read_csv(file_path)
        data_frames.append(df)
    
    combined_data = pd.concat(data_frames, ignore_index=True)
    return combined_data


def combine_days_atc_for_train_all(exp_type="A"):
    hour_groups = {hour: list(range(0, hour + 200, 200)) for hour in range(0, 2000, 200)}
    
    print(hour_groups)

    # Process each group, combine the data, and save to new CSV files
    for hour, hours in hour_groups.items():
        combined_df = combine_data(hours, exp_type)
        combined_df.to_csv(f'thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/combine_for_train_all/{exp_type}_magni_combine_{hour}_{hour+200}.csv', index=False)
    
os.makedirs('thor_magni_combine_add_time_all_dates_for_train_cliff_correct_ori/combine_for_train_all', exist_ok=True)
combine_days_atc_for_train_all(exp_type='A')