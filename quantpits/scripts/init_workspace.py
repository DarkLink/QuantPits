#!/usr/bin/env python3
"""
Initialize a new Qlib Workspace (Pit)
Usage: python quantpits/scripts/init_workspace.py --source workspaces/Demo_Workspace --target ~/.quantpits/workspaces/MyNewWorkspace
"""
import argparse
import os
import platform
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
        
    # 1.5 Generate strategy_config.yaml if missing
    target_strategy_yaml = os.path.join(target_config, "strategy_config.yaml")
    if not os.path.exists(target_strategy_yaml):
        print(f"Generating default strategy_config.yaml at {target_strategy_yaml}")
        default_yaml = '''# Strategy Provider Configuration
strategy:
  name: topk_dropout
  params:
    topk: 20
    n_drop: 3
    only_tradable: true
    buy_suggestion_factor: 2

backtest:
  account: 100000000
  exchange_kwargs:
    limit_threshold: 0.095
    deal_price: close
    open_cost: 0.0005
    close_cost: 0.0015
    min_cost: 5
'''
        with open(target_strategy_yaml, "w") as f:
            f.write(default_yaml)
            
    # 2. Create empty data, output, archive directories
    for d in ["data", "output", "archive", "mlruns"]:
        dir_path = os.path.join(target, d)
        print(f"Creating empty directory: {dir_path}")
        os.makedirs(dir_path)
    # 3. Create activation script
    is_windows = platform.system() == "Windows"
    script_ext = ".ps1" if is_windows else ".sh"
    script_name = "run_env" + script_ext
    run_env_path = os.path.join(target, script_name)

    with open(run_env_path, "w", encoding="utf-8") as f:
        if is_windows:
            f.write("# run this file to activate the workspace\n\n")
            f.write('$env:QLIB_WORKSPACE_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path\n')
            f.write("# Uncomment and modify to use a custom Qlib data directory:\n")
            f.write('# $env:QLIB_DATA_DIR = "D:\\data\\cn_data"\n')
            f.write('# $env:QLIB_REGION = "cn"\n\n')
            f.write('Write-Host "Workspace activated: $env:QLIB_WORKSPACE_DIR"\n')
        else:
            f.write("#!/bin/bash\n")
            f.write("# Source this file to activate the workspace\n")
            f.write('export QLIB_WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n')
            f.write("# Uncomment and modify to use a custom Qlib data directory:\n")
            f.write('# export QLIB_DATA_DIR="~/.qlib/qlib_data/cn_data"\n')
            f.write('# export QLIB_REGION="cn"\n')
            f.write('echo "Workspace activated: $QLIB_WORKSPACE_DIR"\n')

    # Make it executable for Linux/macOS
    if not is_windows:
        os.chmod(run_env_path, os.stat(run_env_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print("\nWorkspace initialization complete!")
    if is_windows:
        print(f"To use this workspace, run: .\\{script_name}")
        print(f"Or in PowerShell: $env:QLIB_WORKSPACE_DIR='{target}'; python ...")
    else:
        print(f"To use this workspace, run: source {run_env_path}")
        print(f"Or prepend commands with: QLIB_WORKSPACE_DIR={target} python ...")
    # 3. Create run_env.sh
    # run_env_path = os.path.join(target, "run_env.sh")
    # with open(run_env_path, "w") as f:
    #     f.write("#!/bin/bash\n")
    #     f.write("# Source this file to activate the workspace\n")
    #     f.write('export QLIB_WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n')
    #     f.write("# Uncomment and modify to use a custom Qlib data directory:\n")
    #     f.write('# export QLIB_DATA_DIR="~/.qlib/qlib_data/cn_data"\n')
    #     f.write('# export QLIB_REGION="cn"\n')
    #     f.write("echo \"Workspace activated: $QLIB_WORKSPACE_DIR\"\n")
        
    # Make it executable just in case, though it should be sourced
    # os.chmod(run_env_path, os.stat(run_env_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    
    # print("\\nWorkspace initialization complete!")
    # print(f"To use this workspace, run: source {run_env_path}")
    # print(f"Or prepend commands with: QLIB_WORKSPACE_DIR={target} python ...")

def main():
    parser = argparse.ArgumentParser(description="Initialize a new Qlib Workspace")
    parser.add_argument("--source", required=True, help="Path to existing workspace to clone config from (e.g., workspaces/Demo_Workspace)")
    parser.add_argument("--target", required=True, help="Path to new workspace directory (e.g., ~/.quantpits/workspaces/MyNewWorkspace)")
    args = parser.parse_args()
    
    init_workspace(args.source, args.target)

if __name__ == "__main__":
    main()
