"""一键运行完整评估流程

执行流程：
1. 运行五线程并行自对弈评估
2. 分析评估结果，生成统计数据
3. 生成可视化图表
4. 创建HTML报告

使用方法：
    python scripts/run_full_evaluation.py --games-per-pair 10 --threads 5
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def run_command(cmd: list, description: str) -> bool:
    """运行命令并返回是否成功
    
    Args:
        cmd: 命令列表
        description: 命令描述
        
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 80)
    print(f" {description}")
    print("=" * 80 + "\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} 失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="一键运行完整评估流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
执行流程:
  1. 五线程并行自对弈评估
  2. 数据分析
  3. 可视化生成
  4. HTML报告生成 (TODO)

示例用法:
  # 使用默认配置
  python scripts/run_full_evaluation.py
  
  # 自定义配置
  python scripts/run_full_evaluation.py --games-per-pair 20 --threads 5
  
  # 快速测试（每配对1局）
  python scripts/run_full_evaluation.py --games-per-pair 1 --threads 2
        """
    )
    
    parser.add_argument(
        '--games-per-pair',
        type=int,
        default=10,
        help='每个配对的游戏数（单向）'
    )
    
    parser.add_argument(
        '--threads',
        type=int,
        default=5,
        help='并行线程数'
    )
    
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='跳过评估，直接分析和可视化现有结果'
    )
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    print("=" * 80)
    print(" 五线程自对弈评估 - 完整流程")
    print("=" * 80)
    print(f"\n开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"每配对游戏数: {args.games_per_pair} × 2 (轮换先后手)")
    print(f"并行线程数: {args.threads}")
    
    # 步骤1: 运行评估
    if not args.skip_evaluation:
        success = run_command(
            [
                sys.executable,
                "scripts/parallel_eval_5threads.py",
                "--games-per-pair", str(args.games_per_pair),
                "--threads", str(args.threads)
            ],
            "步骤 1/3: 五线程并行评估"
        )
        
        if not success:
            print("\n❌ 评估失败，流程终止")
            return
    else:
        print("\n⏭️  跳过评估步骤")
    
    # 步骤2: 数据分析
    success = run_command(
        [
            sys.executable,
            "scripts/analyze_results.py"
        ],
        "步骤 2/3: 数据分析"
    )
    
    if not success:
        print("\n⚠️  数据分析失败，但继续尝试可视化")
    
    # 步骤3: 生成可视化
    success = run_command(
        [
            sys.executable,
            "scripts/generate_visualizations.py"
        ],
        "步骤 3/3: 生成可视化"
    )
    
    if not success:
        print("\n⚠️  可视化生成失败")
    
    # 完成
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print(" 完整评估流程完成")
    print("=" * 80)
    print(f"\n开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    
    print("\n📁 结果位置:")
    print("  - 评估结果: ./data/results/self_play/5threads/")
    print("  - 统计数据: ./data/results/")
    print("  - 可视化图表: ./data/results/visualizations/")
    
    print("\n🎯 下一步:")
    print("  1. 查看可视化图表: ./data/results/visualizations/")
    print("  2. 查看统计报告: ./data/results/self_play/5threads/statistics_*.txt")


if __name__ == "__main__":
    main()
