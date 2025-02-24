

from datetime import datetime, timedelta
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# all_time = [
#     "12:17",
#     "12:20",
#     "12:28",
#     "12:39",
#     "12:53",
#     "13:13",
#     "13:37",
#     "14:06",
#     "14:47",
#     "15:35",
#     "16:25",
#     "17:32",
#     "18:52",
#     "20:20",
#     "21:55",
#     "23:37",
#     "01:32",
#     "03:35",
#     "05:52",
#     "08:26",
#     "11:06",
# ]

# interval_time = [
#     "12:17",
#     "12:20",
#     "12:22",
#     "12:24",
#     "12:25",
#     "12:27",
#     "12:28",
#     "12:30",
#     "12:31",
#     "12:33",
#     "12:34",
#     "12:37",
#     "12:39",
#     "12:41",
#     "12:43",
#     "12:45",
#     "12:47",
#     "12:49",
#     "12:50",
#     "12:52",
#     "12:53",    
# ]

# online_time = [
#     185.60,
#     4.42,
#     7.36,
#     4.37,
#     4.30,
#     3.04,
#     2.88,
#     2.79,
#     2.95,
#     1.90,
#     9.58,
#     3.56,
#     2.79,
#     2.79,
#     2.79,
#     9.86,
#     2.82,
#     2.88,
#     3.08,
#     6.54]

# def convert_to_mins(all_time):
#     # Convert time strings to datetime objects, assuming the first time is on day 1
#     time_objects = [datetime.strptime(t, "%H:%M") for t in all_time]
#     for i in range(1, len(time_objects)):
#         # Check if the time rolls over to the next day
#         if time_objects[i] < time_objects[i - 1]:
#             time_objects[i] += timedelta(days=1)  # Increment the day by 1 if there is a rollover

#     # Calculate intervals in minutes
#     intervals = [(time_objects[i] - time_objects[i - 1]).total_seconds() / 60 for i in range(1, len(time_objects))]

#     return intervals

# all_time_intervals = convert_to_mins(all_time)
# interval_time_intervals = convert_to_mins(interval_time)
# online_time_intervals = [round(seconds / 60, 3) for seconds in online_time]

# print(f"all_time_intervals: {all_time_intervals}")
# print(f"interval_time_intervals: {interval_time_intervals}")
# print(f"online_time_intervals: {online_time_intervals}")


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

data = {
    'Sequence': range(1, 6),
    'History': all_time_intervals[:5],
    'Interval': interval_time_intervals[:5],
    'Online': online_time_intervals[:5]
}
df = pd.DataFrame(data)

# Melting the DataFrame to 'long format'
df_long = df.melt('Sequence', var_name='Type', value_name='Time (mins)')

# Creating the plot with Seaborn
plt.figure(figsize=(9, 4))
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
plt.xlabel('Iteration Number', fontsize=20)
plt.ylabel('Time (minutes)', fontsize=20)
legend = plt.legend(fontsize='x-large')  # You can specify 'small', 'medium', 'large', 'x-large', etc.
for handle in legend.legendHandles:
    handle.set_markersize(10)  # Set a larger size for markers
    handle.set_marker('o')
plt.xticks(range(1, 6), fontsize=20)
plt.yticks(fontsize=20)
# plt.grid(True)
# plt.show()
plt.savefig('runningtime_magni.pdf')