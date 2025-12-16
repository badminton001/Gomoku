import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def extract_latest_data(log_dir):
    """Extract latest training data from log directory"""
    if not os.path.exists(log_dir):
        print(f"Error: Log directory '{log_dir}' not found.")
        return None

    runs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]

    if not runs:
        print(f"Error: No logs found in {log_dir}")
        return None

    latest_run = max(runs, key=os.path.getmtime)
    print(f"Reading logs from: {latest_run}")

    event_acc = EventAccumulator(latest_run)
    event_acc.Reload()

    tags = event_acc.Tags()['scalars']
    data = {}

    keys = ['rollout/ep_rew_mean', 'rollout/ep_len_mean', 'train/loss']

    for k in keys:
        if k in tags:
            events = event_acc.Scalars(k)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            data[k] = pd.DataFrame({'step': steps, 'value': values})

    return data


def plot_15x15_performance(data):
    """Plot training performance chart"""
    if not data:
        print("No data to plot.")
        return

    plt.figure(figsize=(15, 5))

    if 'rollout/ep_rew_mean' in data:
        df = data['rollout/ep_rew_mean']
        plt.subplot(1, 3, 1)
        plt.plot(df['step'], df['value'], color='blue', linewidth=2, label='Reward')
        plt.xlabel('Timesteps')
        plt.ylabel('Average Reward')
        plt.title('Reward Curve')
        plt.grid(True, alpha=0.3)
        plt.legend()

    if 'rollout/ep_len_mean' in data:
        df = data['rollout/ep_len_mean']
        plt.subplot(1, 3, 2)
        plt.plot(df['step'], df['value'], color='green', linewidth=2, label='Avg Steps')
        plt.xlabel('Timesteps')
        plt.ylabel('Steps per Game')
        plt.title('Game Length')
        plt.grid(True, alpha=0.3)
        plt.legend()

    if 'train/loss' in data:
        df = data['train/loss']
        plt.subplot(1, 3, 3)
        plt.plot(df['step'], df['value'], color='orange', linewidth=1, label='Loss')
        plt.xlabel('Timesteps')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()

    output_path = 'data/charts/dqn_v1_training_curve.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"[OK] Plot saved to {output_path}")


if __name__ == "__main__":
    log_dir = "./data/logs/dqn_15x15_final_tensorboard/"

    data = extract_latest_data(log_dir)
    plot_15x15_performance(data)