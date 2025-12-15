import os
import sys
import subprocess
import shutil
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def install_dependencies():
    print(">>> Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "stable-baselines3", "shimmy", "tensorboard", "pandas", "matplotlib", "seaborn"])
    except Exception as e:
        print(f"Warning: Dependency installation failed: {e}")

def find_script(script_name):
    # Search for script in /kaggle/working
    for root, dirs, files in os.walk("/kaggle/working"):
        if script_name in files:
            return os.path.join(root, script_name)
    return None

def run_evaluation():
    print(">>> Locating Evaluation Script...")
    script_path = find_script("run_full_evaluation.py")
    
    if not script_path:
        print("❌ Error: Could not find 'run_full_evaluation.py' anywhere!")
        return
        
    print(f"✅ Found script at: {script_path}")
    
    # Calculate Project Root (Parent of 'scripts')
    # script_path is .../scripts/run_full_evaluation.py
    # we want .../ (Parent of scripts)
    scripts_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(scripts_dir)
    
    print(f"✅ Project Root appears to be: {project_root}")
    
    # Change CWD to Project Root so imports work
    os.chdir(project_root)
    print(f"working directory changed to: {os.getcwd()}")
    
    # Run
    cmd = [sys.executable, script_path, "--games-per-pair", "10", "--threads", "4"]
    subprocess.check_call(cmd)

def package_results():
    print(">>> Packaging Results...")
    
    # Results should be in ./data/results relative to CWD
    src_dir = "./data/results"
    
    if not os.path.exists(src_dir):
        print(f"Error: Results directory {src_dir} not found in {os.getcwd()}!")
        # Try to find it if it's elsewhere
        for root, dirs, files in os.walk("/kaggle/working"):
            if "results" in dirs and "data" in root.split(os.sep):
                 src_dir = os.path.join(root, "results")
                 print(f"Found it at: {src_dir}")
                 break
    
    if os.path.exists(src_dir):
        # Create zip in /kaggle/working so it's downloadable
        output_path = "/kaggle/working/gomoku_results"
        shutil.make_archive(output_path, 'zip', src_dir)
        print(f"✅ Success! Results packed to: {output_path}.zip")
    else:
        print("❌ Still could not find results.")

def main():
    print("=== Kaggle Evaluation & Report Generator ===")
    
    # 1. Install Libs (SB3 is likely missing on standard Kaggle image)
    install_dependencies()
    
    # 2. Run Eval
    try:
        run_evaluation()
    except Exception as e:
        print(f"Evaluation Failed: {e}")
        # We still try to package whatever logs exist
        
    # 3. Zip Output
    package_results()
    
    print("\n=== DONE ===")
    print("Please download 'gomoku_results.zip' from the Output tab.")

if __name__ == "__main__":
    main()
