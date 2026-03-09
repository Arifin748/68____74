import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("results/RCPO-{SafetyPointGoal1-v0}/seed-000-2026-03-09-16-07-29/progress.csv")

steps = data["TotalEnvSteps"]

plt.plot(steps, data["Metrics/EpRet"], label="Reward")
plt.plot(steps, data["Metrics/EpCost"], label="Cost")

plt.xlabel("Environment Steps")
plt.ylabel("Value")
plt.title("RCPO Training")

plt.legend()
plt.show()

