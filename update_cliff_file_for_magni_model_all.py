import pandas as pd
import os

from pprint import pprint

# Directory containing the CSV files
exp_first = 'A'
exp_type = 'online'

directory = f"online_mod_res_magni_{exp_first}_first_split_random_{exp_type}/"

range_values = list(range(0, 2001, 200))

if exp_first == 'A':
    file_prefixes = ['A', 'B']
elif exp_first == 'B':
    file_prefixes = ['B', 'A']
        
# Function to generate file names
def generate_filenames():
    for prefix in file_prefixes:
        for start in range_values[:-1]:  # Skip the last since it does not form a valid range
            yield f"{prefix}_{start}_{start + 200}_{exp_type}.csv"

# Read the first file and set it as the baseline
filenames = list(generate_filenames())

print(filenames)

base_df = pd.read_csv(os.path.join(directory, filenames[0]), header=None)
base_df.columns = ["x", "y", "motion_angle", "velocity", "cov1", "cov2", "cov3", "cov4", "weight", "motion_ratio"]
unique_locations = set(zip(base_df['x'], base_df['y']))

new_directory = f"online_mod_res_magni_{exp_first}_first_split_random_{exp_type}_updated/"
os.makedirs(new_directory, exist_ok=True)
print(os.path.join(new_directory, f"{filenames[0]}"))


# Save the first file unchanged in the new location
base_df.to_csv(os.path.join(new_directory, f"{filenames[0]}"), index=False, header=False)

# Process the rest of the files
for filename in filenames[1:]:
    current_df = pd.read_csv(os.path.join(directory, filename), header=None)
    current_df.columns = ["x", "y", "motion_angle", "velocity", "cov1", "cov2", "cov3", "cov4", "weight", "motion_ratio"]
    current_locations = set(zip(current_df['x'], current_df['y']))
    
    # Determine new rows to add from previous locations not in current file
    missing_locations = unique_locations - current_locations
    
    rows_to_add = base_df[base_df.apply(lambda row: (row['x'], row['y']) in missing_locations, axis=1)]
        
    # Combine current DataFrame with the rows to add
    updated_df = pd.concat([current_df, rows_to_add])
    
    # Save the updated DataFrame
    updated_df.to_csv(os.path.join(new_directory, f"{filename}"), index=False, header=False)
    
    base_df = updated_df
    # Update the unique locations set
    unique_locations.update(current_locations)
