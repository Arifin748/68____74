import omnisafe

env_id = "SafetyPointGoal1-v0"

seeds = [0, 1, 2, 3, 4]

for seed in seeds:

    custom_cfgs = {
        "seed": seed,
        "train_cfgs": {
            "total_steps": 200000
        },
        "logger_cfgs": {
            "use_tensorboard": False,
            "use_wandb": False,
            "log_dir": "results"
        }
    }

    print(f"Running seed {seed}")

    agent = omnisafe.Agent(
        algo="RCPO",
        env_id=env_id,
        custom_cfgs=custom_cfgs
    )

    agent.learn()