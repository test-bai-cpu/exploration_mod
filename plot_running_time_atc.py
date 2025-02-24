

from datetime import datetime, timedelta
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np

all_time = [
    "Jun 5 14:37",
    "Jun 5 14:45",
    "Jun 5 16:22",
    "Jun 5 21:45",
    "Jun 6 12:23",
    "Jun 7 18:31",
    "Jun 9 10:27",
    "Jun 11 15:03",
]

# all_time = [
#     "18:01",
#     "18:09",
#     "19:49",
#     "02:12",
#     "11:46",
#     "01:10",
#     "13:20",
#     "03:42",
#     "20:19",
#     "21:53",
#     "23:12",
#     "16:04",
#     "21:40",
# ]

all_time_in_seconds = [
    563.39,
    6484.55,
    29470.78,
    63899.68,
    112165.99,
    155974.02,
    207697.75,
    267489.45,
    359472.13,
    450613.80,
    511331.64,
    531486.62,
]

interval_time = [
    "Jun 5 14:37",
    "Jun 5 14:45",
    "Jun 5 15:52",
    "Jun 5 17:27",
    "Jun 5 21:15",
    "Jun 6 01:18",
    "Jun 6 03:53",
    "Jun 6 07:14",
    "Jun 6 10:32",
    "Jun 6 14:56",
    "Jun 6 19:01",
    "Jun 6 20:49",
    "Jun 6 21:08",
]

online_time = [
    "14:37",
    "14:45",
    "14:52",
    "15:00",
    "15:04",
    "15:05",
    "15:06",
    "15:07",
    "15:08",
    "15:09",
    "15:10",
    "15:10",
    "15:10",
]

window_time = [
    "Aug 6 01:03",
    "Aug 6 02:13",
    "Aug 6 03:23",
    "Aug 6 06:13",
    "Aug 6 12:03",
    "Aug 6 20:58",
    "Aug 7 03:57",
    "Aug 7 09:39",
    "Aug 7 16:59",
    "Aug 8 03:07",
    "Aug 8 10:54",
    "Aug 8 14:53",
    "Aug 8 16:12",
]

def convert_to_mins(all_time, add_day_list=[]):
    # Convert time strings to datetime objects, assuming the first time is on day 1
    time_objects = [datetime.strptime(t, "%H:%M") for t in all_time]
    for i in range(1, len(time_objects)):
        # Check if the time rolls over to the next day
        if time_objects[i] < time_objects[i - 1]:
            time_objects[i] += timedelta(days=1)  # Increment the day by 1 if there is a rollover

    for day in add_day_list:
        time_objects[day] += timedelta(days=1)
    # Calculate intervals in minutes
    intervals = [(time_objects[i] - time_objects[i - 1]).total_seconds() / 60 for i in range(1, len(time_objects))]

    return intervals

# all_time_intervals = convert_to_mins(all_time, add_day_list=[8, 9])
# interval_time_intervals = convert_to_mins(interval_time)
# online_time_intervals = convert_to_mins(online_time)
# window_time_intervals = convert_to_mins(window_time)

# print(f"all_time_intervals: {all_time_intervals}")
# print(f"interval_time_intervals: {interval_time_intervals}")
# print(f"online_time_intervals: {online_time_intervals}")
# print(f"window_time_intervals: {window_time_intervals}")


# plt.figure(figsize=(12, 6))
# x_axis = range(1, 21)  # Common x-axis values based on the length of the lists
# plt.plot(x_axis, all_time_intervals, marker='o', label='All Time Intervals (mins)')
# plt.plot(x_axis, interval_time_intervals, marker='x', linestyle='--', label='Interval Time Intervals (mins)')
# plt.plot(x_axis, online_time_intervals, marker='s', linestyle='-.', label='Online Time (mins)')

# plt.title('Comparison of Time Intervals and Online Durations')
# plt.xlabel('Sequence Number')
# plt.ylabel('Time (minutes)')
# plt.legend()
# plt.grid(True)

# # Show the plot
# plt.show()

##########################################








# # # #### Restart plotting in rebuttal ####
# time_dates = [datetime.strptime(time, "%b %d %H:%M") for time in interval_time]

# # Calculate the differences
# time_diffs = [time_dates[i] - time_dates[i - 1] for i in range(1, len(time_dates))]

# # Display the results
# time_diffs_minutes = [diff.total_seconds() / 60 for diff in time_diffs]  # Convert timedelta to minutes
# # time_diffs_minutes = [diff.total_seconds() for diff in time_diffs]  # Convert timedelta to minutes
# print(time_diffs_minutes)



# all_time_in_minutes = [time / 60 for time in all_time_in_seconds]
# # all_time_in_minutes = [time for time in all_time_in_seconds]

# test_min = []
# for i, time in enumerate(all_time_in_minutes, start=1):
#     # print(f"Time {i}: {time:.2f} minutes")
#     test_min.append(np.round(time, 2))
    
# print(test_min)



# window
window_minute = [100.0, 100.0, 170.0, 350.0, 535.0, 419.0, 342.0, 440.0, 608.0, 467.0, 239.0, 79.0]

# all
all_minute = [8.0, 108.08, 491.18, 1064.99, 1869.43, 2599.57, 3461.63, 4458.16, 5991.2, 7510.23, 8522.19, 8858.11]

# interval
interval_minute = [8.0, 67.0, 95.0, 228.0, 243.0, 155.0, 201.0, 198.0, 264.0, 245.0, 108.0, 19.0]

# online
# online_minute = [8.0, 7.0, 8.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
online_minute = [8.0, 7.0, 8.0, 4.0, 1.0, 38.0/60, 43.1/60, 41.2/60, 54.2/60, 36.6/60, 21.6/60, 9.3/60]


# STeF
# 48.89493465423584

# for list in [window_minute, all_minute, interval_minute, online_second]:
for list in [online_minute, interval_minute, window_minute, all_minute]:
    print(np.round(np.mean(list)))
    # print(len(list))



data = {
    'Sequence': range(9, 21),
    'History': all_minute,
    # 'Window': window_minute,
    'Interval': interval_minute,
    'Online': online_minute
}
df = pd.DataFrame(data)

# Melting the DataFrame to 'long format'
df_long = df.melt('Sequence', var_name='Type', value_name='Time (mins)')

plt.figure(figsize=(10, 6))
ax = sns.lineplot(
    data=df_long, 
    x='Sequence', 
    y='Time (mins)', 
    hue='Type', 
    marker='o',
    # style='Type',
    markersize=20,  # Increase marker size
    linewidth=5   # Increase line width
)
ax.set_yscale('log')

handles, labels = ax.get_legend_handles_labels()
new_handles = [plt.Line2D([], [], marker='o', linestyle='-', color=handle.get_color(), markersize=15, linewidth=5) for handle in handles]
ax.legend(handles=new_handles, labels=labels, title='', fontsize='20', title_fontsize='13', loc='upper left')



# plt.xlabel('Iteration Number', fontsize=20)
plt.xlabel('Hour', fontsize=20)
plt.ylabel('Time (minutes)', fontsize=20)
# legend = plt.legend(fontsize='x-large')  # You can specify 'small', 'medium', 'large', 'x-large', etc.
# for handle in legend.legendHandles:
#     handle.set_markersize(10)  # Set a larger size for markers
#     handle.set_marker('o')
plt.xticks(range(9, 21), fontsize=20)
plt.yticks(fontsize=20)
# plt.grid(True)
# plt.show()
plt.savefig('runningtime_atc_full.pdf')
plt.savefig('runningtime_atc_full.png')