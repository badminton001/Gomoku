import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_all_runs(log_dir):
    """读取目录下所有 runs 的数据"""
    if not os.path.exists(log_dir):
        print(f"Error: Log directory '{log_dir}' not found.")
        return {}

    runs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    runs.sort()

    all_data = {}

    for run_name in runs:
        run_path = os.path.join(log_dir, run_name)
        print(f"Reading logs from: {run_name}")

        try:
            event_acc = EventAccumulator(run_path)
            event_acc.Reload()
            tags = event_acc.Tags()['scalars']

            if 'rollout/ep_rew_mean' in tags:
                events = event_acc.Scalars('rollout/ep_rew_mean')
                steps = [e.step for e in events]
                values = [e.value for e in events]
                df = pd.DataFrame({'step': steps, 'value': values})

                if "DQN_1" in run_name:
                    print(f"  -> Truncating {run_name} at 3,300,000 steps...")
                    df = df[df['step'] <= 3_300_000]

                all_data[run_name] = df
        except Exception as e:
            print(f"Skipping {run_name}: {e}")

    return all_data


def plot_comparison(all_data):
    """绘制对比曲线"""
    if not all_data:
        print("No data found to plot.")
        return

    plt.figure(figsize=(10, 6))

    styles = {
        'DQN_1': {'color': 'green', 'label': 'Baseline (Sparse Reward)', 'linestyle': '-'},
        'DQN_2': {'color': 'blue', 'label': 'Ours (Heuristic Reward)', 'linestyle': '-'}
    }

    for run_name, df in all_data.items():
        style = styles.get(run_name, {'label': run_name})

        plt.plot(df['step'], df['value'],
                 label=style['label'],
                 color=style.get('color'),
                 linestyle=style.get('linestyle'),
                 linewidth=2,
                 alpha=0.8)

    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('Performance Comparison: Sparse vs Shaped Reward', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper left')

    plt.tight_layout()

    output_path = 'data/results/comparison_curve.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Comparison plot saved to {output_path}")


if __name__ == "__main__":
    log_dir = "./data/logs/dqn_15x15_tensorboard/"
    all_data = extract_all_runs(log_dir)
    plot_comparison(all_data)