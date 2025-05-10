import pandas as pd
import matplotlib.pyplot as plt

# Load the first and last CSV file
first_file = "dqn_conn0_ep1.csv"
last_file = "dqn_conn0_ep250.csv"

# Read the CSV files
df_first = pd.read_csv(first_file)
df_last = pd.read_csv(last_file)

# Ensure step range and row count are correct
assert df_first["step"].min() == 0.0 and df_first["step"].max() == 1000.0 and df_first.shape[0] == 201
assert df_last["step"].min() == 0.0 and df_last["step"].max() == 1000.0 and df_last.shape[0] == 201

# Plotting for the first and last CSV files
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot system_mean_speed and system_mean_waiting_time for the first file (initial episode)
ax1.plot(df_first['step'], df_first['system_mean_speed'], label="Mean Speed (First Episode)", color='C0', alpha=0.6)
ax1.plot(df_first['step'], df_first['system_mean_waiting_time'], label="Mean Waiting Time (First Episode)", color='C1', alpha=0.6)

# Plot system_mean_speed and system_mean_waiting_time for the last file (final episode)
ax1.plot(df_last['step'], df_last['system_mean_speed'], label="Mean Speed (Last Episode)", color='C0', linestyle='--', linewidth=2)
ax1.plot(df_last['step'], df_last['system_mean_waiting_time'], label="Mean Waiting Time (Last Episode)", color='C1', linestyle='--', linewidth=2)

ax1.set_xlabel("Step")
ax1.set_ylabel("Speed / Waiting Time")
ax1.set_title("DQN Performance: First vs Last Episode")
ax1.legend(loc="upper right")

# Plot system_total_waiting_time for both first and last CSV files
ax2 = ax1.twinx()
ax2.plot(df_first['step'], df_first['system_total_waiting_time'], label="Total Waiting Time (First Episode)", color='C2', alpha=0.6)
ax2.plot(df_last['step'], df_last['system_total_waiting_time'], label="Total Waiting Time (Last Episode)", color='C2', linestyle='--', linewidth=2)
ax2.set_ylabel("Total Waiting Time")

# Show the plot
plt.tight_layout()
plt.show()
