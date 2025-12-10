import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG_DIR = "./data/logs/benchmark_tensorboard/"
RESULTS_DIR = "./data/results/benchmark_plots/"
os.makedirs(RESULTS_DIR, exist_ok=True)


def extract_run_data(run_path):
    """从 TensorBoard 日志提取标量数据"""
    try:
        event_acc = EventAccumulator(run_path)
        event_acc.Reload()
        tags = event_acc.Tags()['scalars']
        data = {}

        keys = ['rollout/ep_rew_mean', 'train/loss', 'rollout/ep_len_mean']

        for k in keys:
            if k in tags:
                events = event_acc.Scalars(k)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                data[k] = pd.DataFrame({'step': steps, 'value': values})
        return data
    except Exception as e:
        print(f"Error reading {run_path}: {e}")
        return None


def find_logs():
    """扫描日志目录并分类"""
    if not os.path.exists(LOG_DIR):
        print(f"Error: Directory {LOG_DIR} not found.")
        return {}

    subdirs = [os.path.join(LOG_DIR, d) for d in os.listdir(LOG_DIR) if os.path.isdir(os.path.join(LOG_DIR, d))]

    categorized = {}

    for path in subdirs:
        name = os.path.basename(path)
        if "Sparse" in name:
            categorized['Sparse'] = path
        elif "Shaped" in name:
            categorized['Shaped'] = path
        elif "Strict" in name:
            categorized['Strict'] = path

    return categorized


def plot_individual(name, data, color, secondary_metric='train/loss'):
    """生成单个实验的性能图"""
    if not data or 'rollout/ep_rew_mean' not in data:
        print(f"Skipping {name}: No reward data.")
        return

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    df = data['rollout/ep_rew_mean']
    plt.plot(df['step'], df['value'], color=color, label=f'{name} Reward')
    plt.xlabel('Timesteps')
    plt.ylabel('Average Reward')
    plt.title(f'{name} Training Performance')
    plt.grid(True, alpha=0.3)
    plt.legend()

    if secondary_metric in data:
        plt.subplot(1, 2, 2)
        df2 = data[secondary_metric]
        label_map = {'train/loss': 'Loss', 'rollout/ep_len_mean': 'Episode Length'}
        col_map = {'train/loss': 'orange', 'rollout/ep_len_mean': 'green'}

        plt.plot(df2['step'], df2['value'], color=col_map.get(secondary_metric, 'red'),
                 label=label_map.get(secondary_metric, 'Metric'))
        plt.xlabel('Timesteps')
        plt.ylabel(label_map.get(secondary_metric, 'Value'))
        plt.title(f'{name} Secondary Metric')
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f"individual_{name.lower()}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved individual plot: {save_path}")
    plt.close()


def plot_comparison(logs_data):
    """生成三种方案的对比图"""
    plt.figure(figsize=(12, 8))

    styles = {
        'Sparse': {'color': 'gray', 'style': '--', 'label': '1. Baseline (Sparse)'},
        'Shaped': {'color': 'orange', 'style': '-.', 'label': '2. Shaped (Heuristic Only)'},
        'Strict': {'color': 'blue', 'style': '-', 'label': '3. Strict (Final Solution)'}
    }

    has_data = False
    for name in ['Sparse', 'Shaped', 'Strict']:
        if name in logs_data and logs_data[name] is not None:
            if 'rollout/ep_rew_mean' in logs_data[name]:
                df = logs_data[name]['rollout/ep_rew_mean']
                df = df[df['step'] <= 1_000_000]

                style = styles[name]
                plt.plot(df['step'], df['value'],
                         color=style['color'],
                         linestyle=style['style'],
                         label=style['label'],
                         linewidth=2)
                has_data = True

    if has_data:
        plt.xlabel('Timesteps', fontsize=12)
        plt.ylabel('Average Reward', fontsize=12)
        plt.title('Comparison of Reward Shaping Strategies', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)

        save_path = os.path.join(RESULTS_DIR, "final_comparison.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved comparison plot: {save_path}")
    else:
        print("No data available for comparison plot.")
    plt.close()


if __name__ == "__main__":
    print(">>> Scanning logs...")
    log_paths = find_logs()

    all_data = {}

    for name, path in log_paths.items():
        print(f"Processing {name} from {path}...")
        data = extract_run_data(path)
        all_data[name] = data

        sec_metric = 'rollout/ep_len_mean' if name == 'Strict' else 'train/loss'
        color = {'Sparse': 'gray', 'Shaped': 'orange', 'Strict': 'blue'}[name]

        plot_individual(name, data, color, sec_metric)

    print("Generating comparison plot...")
    plot_comparison(all_data)

    print("\n✅ All plots generated in ./data/results/benchmark_plots/")