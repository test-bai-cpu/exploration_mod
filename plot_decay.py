

from datetime import datetime, timedelta
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def convert_to_mins(all_time):
    # Convert time strings to datetime objects, assuming the first time is on day 1
    time_objects = [datetime.strptime(t, "%H:%M") for t in all_time]
    for i in range(1, len(time_objects)):
        # Check if the time rolls over to the next day
        if time_objects[i] < time_objects[i - 1]:
            time_objects[i] += timedelta(days=1)  # Increment the day by 1 if there is a rollover

    # Calculate intervals in minutes
    intervals = [(time_objects[i] - time_objects[i - 1]).total_seconds() / 60 for i in range(1, len(time_objects))]

    return intervals

decay_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
nll = [1.86, 1.72, 1.67, 1.65, 1.64, 1.67, 1.70, 1.74, 1.79]
# nll = [2.182, 1.973, 1.853, 1.786, 1.779, 1.757, 1.747, 1.748, 1.752]

nll_2 = []

data = {
    'Sequence': decay_rate,
    'NLL': nll,
}
df = pd.DataFrame(data)

# Melting the DataFrame to 'long format'
# df_long = df.melt('Sequence', var_name='Type', value_name='NLL')

# Creating the plot with Seaborn
plt.figure(figsize=(9, 4))
ax = sns.lineplot(
    data=df, 
    x='Sequence', 
    y='NLL',  
    marker='o',
    # style='Type',
    markersize=20,  # Increase marker size
    linewidth=5   # Increase line width
)
# ax.set_yscale('log')
plt.xlabel('Decay rate', fontsize=20)
plt.ylabel('NLL', fontsize=20)
# legend = plt.legend(fontsize='x-large')  # You can specify 'small', 'medium', 'large', 'x-large', etc.
# for handle in legend.legendHandles:
#     handle.set_markersize(10)  # Set a larger size for markers
#     handle.set_marker('o')
plt.xticks(fontsize=20)
# make yticks to .2f
plt.yticks()
plt.yticks(fontsize=20)
# plt.grid(True)
# plt.show()
plt.savefig('decay_magni2.png')