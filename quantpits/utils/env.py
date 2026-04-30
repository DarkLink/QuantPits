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
    raise RuntimeError("Please source a workspace run_env.sh first!")

# 确保 MLflow 的实验数据 (mlruns) 也隔离存放在当前 Workspace 下
# 使用绝对路径确保无论从哪里执行，路径都指向正确位置
mlruns_dir = os.path.abspath(os.path.join(ROOT_DIR, 'mlruns'))
os.environ["MLFLOW_TRACKING_URI"] = f"file://{mlruns_dir}"

# Qlib 数据路径：优先环境变量，兜底默认值
QLIB_DATA_DIR = os.environ.get("QLIB_DATA_DIR", "~/.qlib/qlib_data/cn_data")
QLIB_REGION = os.environ.get("QLIB_REGION", "cn")


_qlib_initialized = False


def init_qlib():
    """
    Initialize Qlib environment once. Subsequent calls are no-ops.

    Data path and region are controlled via environment variables:
      - QLIB_DATA_DIR: Qlib data dir (default ~/.qlib/qlib_data/cn_data)
      - QLIB_REGION:   Qlib region     (default cn)

    Set these in each workspace's run_env.sh to separate data.
    """
    global _qlib_initialized
    if _qlib_initialized:
        return

    import qlib
    from qlib.constant import REG_CN, REG_US

    region_map = {"cn": REG_CN, "us": REG_US}
    region = region_map.get(QLIB_REGION.lower(), REG_CN)

    qlib.init(provider_uri=QLIB_DATA_DIR, region=region)
    _qlib_initialized = True


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
    print(f"Qlib Data Dir   : {QLIB_DATA_DIR}")
    print(f"Qlib Region     : {QLIB_REGION}")
    print("=" * 60)
    print("Please confirm. (Press Ctrl+C within 3 seconds to abort if this is the wrong workspace!)")
    time.sleep(3)
    print("Executing...")
