import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_tensorboard_data(log_dir):
    """从 TensorBoard 日志提取数据"""
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
    
    keys_to_plot = ['rollout/ep_rew_mean', 'train/loss']
    
    for key in keys_to_plot:
        if key in tags:
            events = event_acc.Scalars(key)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            data[key] = pd.DataFrame({'step': steps, 'value': values})
    
    return data


def plot_curves(data):
    """绘制并保存训练曲线"""
    if not data:
        return

    plt.figure(figsize=(12, 5))

    if 'rollout/ep_rew_mean' in data:
        df = data['rollout/ep_rew_mean']
        plt.subplot(1, 2, 1)
        plt.plot(df['step'], df['value'], color='blue', label='Mean Reward')
        plt.xlabel('Timesteps')
        plt.ylabel('Average Reward')
        plt.title('Training Reward Curve')
        plt.grid(True, alpha=0.3)
        plt.legend()

    if 'train/loss' in data:
        df = data['train/loss']
        plt.subplot(1, 2, 2)
        plt.plot(df['step'], df['value'], color='red', label='Loss')
        plt.xlabel('Timesteps')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()

    output_path = 'data/results/training_curve.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    log_dir = "./data/logs/dqn_15x15_tensorboard/"
    if os.path.exists(log_dir):
        data = extract_tensorboard_data(log_dir)
        plot_curves(data)
    else:
        print(f"Log directory {log_dir} does not exist. Please check your training script settings.")