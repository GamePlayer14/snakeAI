import pandas as pd

class RewardLogger:
    def __init__(self):
        self.step_logs = []

    def log_step(self, step, distance_delta, distance_reward, apple_reward, total_reward):
        self.step_logs.append({
            "Step": step,
            "Distance Delta": distance_delta,
            "Distance Reward": distance_reward,
            "Apple Reward": apple_reward,
            "Total Step Reward": total_reward
        })

    def to_dataframe(self):
        return pd.DataFrame(self.step_logs)

    def clear(self):
        self.step_logs = []
