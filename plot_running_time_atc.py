

from datetime import datetime, timedelta
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

all_time = [
    "14:37",
    "14:45",
    "16:22",
    "21:45",
    "12:23",
    "18:20",
]

interval_time = [
    "14:37",
    "14:45",
    "15:52",
    "17:27",
    "21:15",
    "01:18",  
]

online_time = [
    "14:37",
    "14:45",
    "14:52",
    "15:00",
    "15:04",
    "15:05",
]

def convert_to_mins(all_time, add_day_for_five=False):
    # Convert time strings to datetime objects, assuming the first time is on day 1
    time_objects = [datetime.strptime(t, "%H:%M") for t in all_time]
    for i in range(1, len(time_objects)):
        # Check if the time rolls over to the next day
        if time_objects[i] < time_objects[i - 1]:
            time_objects[i] += timedelta(days=1)  # Increment the day by 1 if there is a rollover

    if add_day_for_five:
        time_objects[5] += timedelta(days=1)
    # Calculate intervals in minutes
    intervals = [(time_objects[i] - time_objects[i - 1]).total_seconds() / 60 for i in range(1, len(time_objects))]

    return intervals

all_time_intervals = convert_to_mins(all_time, add_day_for_five=True)
interval_time_intervals = convert_to_mins(interval_time)
online_time_intervals = convert_to_mins(online_time)

print(f"all_time_intervals: {all_time_intervals}")
print(f"interval_time_intervals: {interval_time_intervals}")
print(f"online_time_intervals: {online_time_intervals}")


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

plt.figure(figsize=(8, 4))
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
plt.savefig('runningtime_atc.pdf')