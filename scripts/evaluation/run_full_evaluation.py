"""Run Full Evaluation Pipeline

Pipeline Steps:
1. Run 5-Process Parallel Self-Play Evaluation
2. Analyze evaluation results and generate statistics
3. Generate visualization charts
4. (Future) Generate HTML report

Usage:
    python scripts/run_full_evaluation.py --games-per-pair 10 --threads 5
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status.
    
    Args:
        cmd: List of command arguments
        description: Description of the command
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n[RUN] {description}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n[OK] {description} Completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {description} Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run Full Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Execution Flow:
  1. 5-Thread Parallel Self-Play Evaluation
  2. Data Analysis
  3. Visualization Generation
  4. (Future) HTML Report Generation

Examples:
  # Default configuration
  python scripts/run_full_evaluation.py
  
  # Custom configuration
  python scripts/run_full_evaluation.py --games-per-pair 20 --threads 5
  
  # Quick test (1 game per pair)
  python scripts/run_full_evaluation.py --games-per-pair 1 --threads 2
        """
    )
    
    parser.add_argument(
        '--games-per-pair',
        type=int,
        default=10,
        help='Games per pair (one way)'
    )
    
    parser.add_argument(
        '--threads',
        type=int,
        default=5,
        help='Number of parallel threads/processes'
    )
    
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation, only analyze and visualize existing results'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/results',
        help='Main output directory (Default: ./data/results)'
    )
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    # Define paths
    base_output = args.output_dir.rstrip("/\\")
    eval_output = f"{base_output}/self_play/5threads"
    
    print("5-Process Self-Play Evaluation - Full Pipeline")
    print(f"\nStart Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Main Output Dir: {base_output}")
    print(f"Games Per Pair: {args.games_per_pair} x 2")
    
    # Step 1: Run Evaluation
    if not args.skip_evaluation:
        success = run_command(
            [
                sys.executable,
                "scripts/evaluation/parallel_eval_5processes.py",
                "--games-per-pair", str(args.games_per_pair),
                "--threads", str(args.threads),
                "--output-dir", eval_output
            ],
            "Step 1/3: Parallel Evaluation"
        )
        
        if not success:
            print("\n[FAILED] Evaluation failed, pipeline terminated.")
            return
    else:
        print("\n[SKIP] Skipping Evaluation Step")
    
    # Step 2: Data Analysis
    success = run_command(
        [
            sys.executable,
            "scripts/analysis/analyze_results.py",
            "--input-dir", eval_output,
            "--output-dir", base_output
        ],
        "Step 2/3: Data Analysis"
    )
    
    if not success:
        print("\n[WARN] Analysis failed, but attempting visualization...")
    
    # Step 3: Generate Visualizations
    success = run_command(
        [
            sys.executable,
            "scripts/analysis/generate_visualizations.py",
            "--input-dir", base_output,
            "--output-dir", f"{base_output}/visualizations"
        ],
        "Step 3/3: Generate Visualizations"
    )
    
    if not success:
        print("\n[WARN] Visualization generation failed")
    
    # Completion
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print("\nFull Pipeline Completed")
    print(f"\nStart Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End Time:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    
    print(f"\nResults Location: {base_output}/")
    print(f"  - Visualizations:  {base_output}/visualizations/")


if __name__ == "__main__":
    main()
