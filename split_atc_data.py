import pandas as pd
import os


def split_atc_data_by_person_id_randomly_once(hour):
    data = pd.read_csv(f'atc-1s-ds-1024-split-hour/atc-1024-{hour}.csv', header=None)
    data.columns = ["time", "person_id", "x", "y", "velocity", "motion_angle"]

    save_folder = "atc-1s-ds-1024-split-hour/split_data_random"
    os.makedirs(save_folder, exist_ok=True)

    # Get unique person_ids and sample 10% of them
    unique_person_ids = data['person_id'].unique()
    test_person_ids = pd.Series(unique_person_ids).sample(frac=0.1, random_state=42).tolist()

    # Split the data based on the sampled person_ids
    test_data = data[data['person_id'].isin(test_person_ids)]
    train_data = data[~data['person_id'].isin(test_person_ids)]

    # Save the test and train datasets to files
    test_filename = f'{save_folder}/atc-1024-{hour}_test.csv'
    train_filename = f'{save_folder}/atc-1024-{hour}_train.csv'

    test_data.to_csv(test_filename, index=False, header=False)
    train_data.to_csv(train_filename, index=False, header=False)


# for hour in range(9, 21, 1):
#     split_atc_data_by_person_id_randomly_once(hour)


def combine_data(hours):
    data_frames = []
    for hour in hours:
        file_path = f'atc-1s-ds-1024-split-hour/split_data_random/atc-1024-{hour}_train.csv'
        df = pd.read_csv(file_path, header=None)
        df.columns = ["time", "person_id", "x", "y", "velocity", "motion_angle"]
        data_frames.append(df)
    
    combined_data = pd.concat(data_frames, ignore_index=True)
    return combined_data


def combine_test_data(hours):
    data_frames = []
    for hour in hours:
        file_path = f'atc-1s-ds-1024-split-hour/split_data_random/atc-1024-{hour}_test.csv'
        df = pd.read_csv(file_path, header=None)
        df.columns = ["time", "person_id", "x", "y", "velocity", "motion_angle"]
        data_frames.append(df)
    
    combined_data = pd.concat(data_frames, ignore_index=True)
    return combined_data


def combine_days_atc_for_train_all():
    hour_groups = {hour: list(range(9, hour + 1)) for hour in range(9, 21)}
    
    print(hour_groups)

    # Process each group, combine the data, 10 for 9 and 10, 11 for 9,10,11, and so on, and save to new CSV files
    for hour, hours in hour_groups.items():
        combined_df = combine_data(hours)
        combined_df.to_csv(f'atc-1s-ds-1024-split-hour/combine_for_train_all/atc_combine_{hour}.csv', index=False, header=False)


def combine_days_atc_for_window_average():
    hour_groups = {hour: list(range(hour - 1, hour + 1)) for hour in range(9, 21)}
    hour_groups[9] = [9, 10]
    # print(hour_groups)

    # Process each group, combine the data, 10 for 9 and 10, 11 for 9,10,11, and so on, and save to new CSV files
    for hour, hours in hour_groups.items():
        combined_df = combine_data(hours)
        combined_df.to_csv(f'atc-1s-ds-1024-split-hour/combine_for_window_average/atc_window2_{hour}.csv', index=False, header=False)
    

# os.makedirs('atc-1s-ds-1024-split-hour/combine_for_train_all', exist_ok=True)
# combine_days_atc_for_train_all()


# os.makedirs('atc-1s-ds-1024-split-hour/combine_for_window_average', exist_ok=True)
# combine_days_atc_for_window_average()



## combine all hours training data together for STeF-map
hours = list(range(9, 21))
combined_df = combine_test_data(hours)
combined_df.to_csv(f'atc-1s-ds-1024-split-hour/atc_combine_test_9_20.csv', index=False, header=False)