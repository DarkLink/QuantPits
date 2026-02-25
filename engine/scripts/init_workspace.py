#!/usr/bin/env python3
"""
Initialize a new Qlib Workspace (Pit)
Usage: python engine/scripts/init_workspace.py --source workspaces/Demo_Workspace --target ~/.quantpits/workspaces/MyNewWorkspace
"""
import argparse
import os
import shutil
import stat

def init_workspace(source, target):
    source = os.path.abspath(source)
    target = os.path.abspath(target)
    
    if not os.path.exists(source):
        print(f"Error: Source workspace '{source}' does not exist.")
        return
        
    if os.path.exists(target):
        print(f"Error: Target workspace '{target}' already exists. Please choose a new directory.")
        return
        
    print(f"Initializing new workspace at: {target}")
    os.makedirs(target)
    
    # 1. Copy config directory
    source_config = os.path.join(source, "config")
    target_config = os.path.join(target, "config")
    if os.path.exists(source_config):
        print(f"Cloning config from {source_config} to {target_config}")
        shutil.copytree(source_config, target_config)
    else:
        print(f"Warning: No config directory found in {source}. Creating empty config.")
        os.makedirs(target_config)
        
    # 2. Create empty data, output, archive directories
    for d in ["data", "output", "archive", "mlruns"]:
        dir_path = os.path.join(target, d)
        print(f"Creating empty directory: {dir_path}")
        os.makedirs(dir_path)
        
    # 3. Create run_env.sh
    run_env_path = os.path.join(target, "run_env.sh")
    with open(run_env_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Source this file to activate the workspace\n")
        f.write('export QLIB_WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n')
        f.write("echo \"Workspace activated: $QLIB_WORKSPACE_DIR\"\n")
        
    # Make it executable just in case, though it should be sourced
    os.chmod(run_env_path, os.stat(run_env_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    
    print("\\nWorkspace initialization complete!")
    print(f"To use this workspace, run: source {run_env_path}")
    print(f"Or prepend commands with: QLIB_WORKSPACE_DIR={target} python ...")

def main():
    parser = argparse.ArgumentParser(description="Initialize a new Qlib Workspace")
    parser.add_argument("--source", required=True, help="Path to existing workspace to clone config from (e.g., workspaces/Demo_Workspace)")
    parser.add_argument("--target", required=True, help="Path to new workspace directory (e.g., ~/.quantpits/workspaces/MyNewWorkspace)")
    args = parser.parse_args()
    
    init_workspace(args.source, args.target)

if __name__ == "__main__":
    main()
