import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 0. Parse optional global --workspace argument
_workspace_arg = None
# We safely iterate without removing it, so upstream argparse won't fail if they don't define it.
# Wait, actually, if upstream uses strict argparse, they might fail on unrecognized args.
# So we need to ensure the entry scripts either ignore it or parse it using parse_known_args.
# But for now, we just read it.
# A better way is to check it and let argparse naturally complain if it's not defined, 
# but we will add it to the argparse of the main scripts where needed, 
# or just remove it from sys.argv so argparse doesn't crash on unrecognized arguments.
for i, arg in enumerate(sys.argv):
    if arg == "--workspace" and i + 1 < len(sys.argv):
        _workspace_arg = sys.argv[i + 1]
        # Remove from sys.argv so downstream argparse doesn't crash on unrecognized arguments
        sys.argv.pop(i) # remove "--workspace"
        sys.argv.pop(i) # remove the path
        break

# 1. 优先使用命令行参数 --workspace
if _workspace_arg:
    ROOT_DIR = os.path.abspath(_workspace_arg)
    os.environ["QLIB_WORKSPACE_DIR"] = ROOT_DIR
# 2. 其次使用环境变量 QLIB_WORKSPACE_DIR
elif "QLIB_WORKSPACE_DIR" in os.environ:
    ROOT_DIR = os.path.abspath(os.environ["QLIB_WORKSPACE_DIR"])
else:
    # 3. 回退到传统的硬编码相对路径 (向后兼容)
    # 如果 env.py 位于 engine/scripts 目录下，ROOT_DIR 应该是 QuantPits 根目录 (即 SCRIPT_DIR 的上两级)
    ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# 确保 MLflow 的实验数据 (mlruns) 也隔离存放在当前 Workspace 下
# 使用绝对路径确保无论从哪里执行，路径都指向正确位置
mlruns_dir = os.path.abspath(os.path.join(ROOT_DIR, 'mlruns'))
os.environ["MLFLOW_TRACKING_URI"] = f"file://{mlruns_dir}"

def safeguard(script_name="Pipeline"):
    """
    Safety lock to prevent accidental execution in the wrong workspace.
    Prints the active workspace and pauses for 3 seconds.
    """
    workspace_name = os.path.basename(ROOT_DIR)
    print("=" * 60)
    print(f"🚦  SAFEGUARD ACTIVATED  🚦")
    print(f"[{script_name}] is about to run.")
    print(f"Active Workspace: \033[1;31;40m{workspace_name}\033[0m")
    print(f"Workspace Path  : {ROOT_DIR}")
    print("=" * 60)
    print("Please confirm. (Press Ctrl+C within 3 seconds to abort if this is the wrong workspace!)")
    time.sleep(3)
    print("Executing...")
