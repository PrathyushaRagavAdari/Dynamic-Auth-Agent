import os
import sys
import json

def run_smoke_test():
    print("Running GraphGuard System Smoke Test...")
    
    # 1. Check Directories
    required_dirs = ["logs", "artifacts", "config"]
    for d in required_dirs:
        if not os.path.exists(d):
            print(f"[FIXING] Missing directory '{d}'. Creating now...")
            os.makedirs(d)
            
    # 2. Check Config
    try:
        with open("config/settings.json", "r") as f:
            config = json.load(f)
            print(f"[PASS] Loaded config for {config['app_name']}")
    except FileNotFoundError:
        print("[FAIL] Missing config/settings.json")
        sys.exit(1)

    # 3. Check Critical Dependencies
    try:
        import streamlit
        import pandas
        print("[PASS] Core dependencies imported successfully.")
    except ImportError as e:
        print(f"[FAIL] Missing dependency: {e}")
        sys.exit(1)

    print("Smoke Test Passed! System is ready.")

if __name__ == "__main__":
    run_smoke_test()