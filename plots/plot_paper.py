import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

# หา progress.csv ทุก seed
files = glob.glob(
    "results/RCPO/SafetyPointGoal1-v0/seed*/progress.csv"
)

rewards = []
costs = []

for f in files:
    data = pd.read_csv(f)

    rewards.append(data["Metrics/EpRet"].values)
    costs.append(data["Metrics/EpCost"].values)

rewards = np.array(rewards)
costs = np.array(costs)

steps = data["TotalEnvSteps"]

# mean/std
reward_mean = rewards.mean(axis=0)
reward_std = rewards.std(axis=0)

cost_mean = costs.mean(axis=0)
cost_std = costs.std(axis=0)

# -------- reward plot --------
plt.figure()

plt.plot(steps, reward_mean, label="RCPO")

plt.fill_between(
    steps,
    reward_mean - reward_std,
    reward_mean + reward_std,
    alpha=0.3
)

plt.xlabel("Environment Steps")
plt.ylabel("Reward")
plt.title("Reward vs Steps")

plt.legend()
plt.show()

# -------- cost plot --------
plt.figure()

plt.plot(steps, cost_mean, label="RCPO")

plt.fill_between(
    steps,
    cost_mean - cost_std,
    cost_mean + cost_std,
    alpha=0.3
)

plt.xlabel("Environment Steps")
plt.ylabel("Cost")
plt.title("Cost vs Steps")

plt.legend()
plt.show()