# Engineering Audit & Stress Test Findings
**Date:** March 2026
**Auditors:** Dynamic Agents Team

## 1. The Audit Process
We simulated a fresh deployment by deleting our virtual environments (`rm -rf venv`), clearing all cached files, and having a teammate clone the repository to a completely clean machine. 

## 2. Breakpoints & Failures Discovered
* **Failure 1: Missing Directories (Crash).** The system immediately crashed with a `FileNotFoundError`. The `app/main.py` attempted to write to `logs/pipeline_logs.csv`, but Git does not track empty folders, so the `logs/` directory did not exist on the fresh clone.
* **Failure 2: Hardcoded Configurations.** We noticed that the target latency metrics and file paths were hardcoded directly inside `app/main.py`, making the system rigid and difficult to deploy to a new environment (like a cloud server with different volume mounts).
* **Failure 3: Unpinned Dependencies.** `requirements.txt` listed packages without version numbers (e.g., `pandas`), risking future breakages if a dependency updated with breaking changes.

## 3. Implemented Fixes
1. **Automated Directory Creation:** Added a `scripts/smoke_test.py` that verifies and creates `logs/`, `config/`, and `artifacts/` directories before the app ever launches.
2. **Config-Driven Execution:** Extracted hardcoded variables into `config/settings.json`.
3. **Dependency Pinning:** Updated `requirements.txt` with strict versioning (`==`).