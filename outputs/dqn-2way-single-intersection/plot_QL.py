import glob, pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load and concatenate
dfs = []
for fn in glob.glob("ql-4x4grid_run1_conn0_ep*.csv"):
    ep = int(fn.split("ep")[1].split(".csv")[0])
    df = pd.read_csv(fn)
    df["episode"] = ep
    dfs.append(df)
data = pd.concat(dfs, ignore_index=True)

# pivot to (steps × episodes) for each metric
def pivot_metric(name):
    return data.pivot(index="step", columns="episode", values=name)

m_speed   = pivot_metric("system_mean_speed")
m_wait    = pivot_metric("system_mean_waiting_time")
t_wait    = pivot_metric("system_total_waiting_time")

# compute mean±std
mean_speed, std_speed = m_speed.mean(1), m_speed.std(1)
mean_mwait, std_mwait = m_wait.mean(1), m_wait.std(1)
mean_twait, std_twait = t_wait.mean(1), t_wait.std(1)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# mean speed & mean waiting time on left axis
ax1.plot(mean_speed.index, mean_speed, label="mean speed")
ax1.fill_between(mean_speed.index, mean_speed-std_speed, mean_speed+std_speed, alpha=0.3)
ax1.plot(mean_mwait.index, mean_mwait, label="mean waiting time")
ax1.fill_between(mean_mwait.index, mean_mwait-std_mwait, mean_mwait+std_mwait, alpha=0.3)

# total waiting time on right axis
ax2.plot(mean_twait.index, mean_twait, label="total waiting time", linestyle="--")
ax2.fill_between(mean_twait.index, mean_twait-std_twait, mean_twait+std_twait, alpha=0.2)

ax1.set_xlabel("step")
ax1.set_ylabel("mean speed / mean waiting time")
ax2.set_ylabel("total waiting time")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.title("Metric trajectories (mean ±1 std over episodes)")
plt.show()
