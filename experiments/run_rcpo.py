import omnisafe

env_id = "SafetyPointGoal1-v0"

custom_cfgs = {
    "train_cfgs": {
        "total_steps": 200000
    },
    "logger_cfgs": {
        "use_tensorboard": True,
        "use_wandb": False,
        "log_dir": "results"
    }
}

agent = omnisafe.Agent(
    algo="RCPO",
    env_id=env_id,
    custom_cfgs=custom_cfgs
)

agent.learn()