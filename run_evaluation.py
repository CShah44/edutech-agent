"""
Quick Start Script for ELI5 Evaluation
=======================================

This script provides easy commands to run evaluations on your GPU server.
"""

import subprocess
import sys
from pathlib import Path

def print_banner():
    print("""
╔════════════════════════════════════════════════════════════╗
║          ELI5 Multi-Agent Evaluation System                ║
║                                                            ║
║  Evaluate your multi-agent system against 325k questions  ║
╚════════════════════════════════════════════════════════════╝
""")

def print_menu():
    print("\n📋 Available Operations:\n")
    print("1. Run Split 1 (rows 0-108k) with config1")
    print("2. Run Split 2 (rows 108k-216k) with config1")
    print("3. Run Split 3 (rows 216k-325k) with config1")
    print("4. Run small test (first 10 questions)")
    print("5. Generate report from results file")
    print("6. Custom command")
    print("0. Exit")
    print()

def run_command(cmd):
    """Run command and display output"""
    print(f"\n🚀 Running: {' '.join(cmd)}\n")
    subprocess.run(cmd)

def main():
    print_banner()
    
    while True:
        print_menu()
        choice = input("Enter your choice (0-6): ").strip()
        
        if choice == '0':
            print("\n👋 Goodbye!")
            break
            
        elif choice == '1':
            run_command([sys.executable, "evaluation.py", "--split", "0", "--config", "config1"])
            
        elif choice == '2':
            run_command([sys.executable, "evaluation.py", "--split", "1", "--config", "config1"])
            
        elif choice == '3':
            run_command([sys.executable, "evaluation.py", "--split", "2", "--config", "config1"])
            
        elif choice == '4':
            run_command([sys.executable, "evaluation.py", "--start", "0", "--end", "10", "--config", "config1"])
            
        elif choice == '5':
            # List available result files
            results_dir = Path("evaluation_results")
            if results_dir.exists():
                result_files = list(results_dir.glob("*.json"))
                result_files = [f for f in result_files if not f.name.endswith('.report.json')]
                
                if result_files:
                    print("\n📁 Available result files:")
                    for i, f in enumerate(result_files, 1):
                        print(f"  {i}. {f.name}")
                    
                    file_choice = input("\nEnter file number: ").strip()
                    try:
                        file_idx = int(file_choice) - 1
                        if 0 <= file_idx < len(result_files):
                            run_command([sys.executable, "evaluation.py", "--report", str(result_files[file_idx])])
                        else:
                            print("❌ Invalid file number")
                    except ValueError:
                        print("❌ Please enter a valid number")
                else:
                    print("\n⚠️  No result files found in evaluation_results/")
            else:
                print("\n⚠️  evaluation_results/ directory not found")
                
        elif choice == '6':
            print("\n📝 Custom command examples:")
            print("  --split 0 --config config2")
            print("  --start 0 --end 500 --config config1")
            print("  --split 1 --resume evaluation_results/eval_config1_split_20251106_120000.json")
            
            custom_args = input("\nEnter arguments: ").strip()
            if custom_args:
                run_command([sys.executable, "evaluation.py"] + custom_args.split())
            
        else:
            print("❌ Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
