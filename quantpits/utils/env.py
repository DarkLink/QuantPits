import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 0. Parse optional global --workspace argument
_workspace_arg = None
# We remove --workspace / <path> from sys.argv so downstream argparse
# in entry-point scripts does not crash on unrecognised arguments.
for i, arg in enumerate(sys.argv):
    if arg == "--workspace" and i + 1 < len(sys.argv):
        _workspace_arg = sys.argv[i + 1]
        sys.argv.pop(i)  # remove "--workspace"
        sys.argv.pop(i)  # remove the path value
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


def _has_legacy_mlruns_data(workspace_root: str) -> bool:
    """Return True if *workspace_root*/mlruns/ contains real experiment data.

    A directory that contains only a ``.gitkeep`` placeholder (or is entirely
    empty) is NOT considered to have legacy data.
    """
    mlruns = os.path.join(workspace_root, "mlruns")
    if not os.path.isdir(mlruns):
        return False
    for entry in os.listdir(mlruns):
        if entry != ".gitkeep":
            return True
    return False


def _resolve_mlflow_backend(workspace_root: str) -> str:
    """Resolve the MLflow tracking backend URI for *workspace_root*.

    Resolution order
    ----------------
    1. If ``MLFLOW_TRACKING_URI`` is already set in the environment, honour it
       (user-supplied value wins — set it in ``run_env.sh`` to override).
    2. If ``<workspace>/mlruns/`` contains real experiment data (legacy
       workspace), fall back to ``file://<workspace>/mlruns`` and emit a
       one-time migration hint to stderr.
    3. Otherwise, use ``sqlite:///<workspace>/mlflow.db`` — the new default
       that works out-of-the-box with mlflow ≥ 3.0.

    Side-effects
    ------------
    * Sets ``os.environ["MLFLOW_TRACKING_URI"]`` to the resolved URI.
    * Sets ``os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"`` whenever the
      resolved backend starts with ``file://`` so that mlflow ≥ 3.0 does not
      raise its maintenance-mode exception.

    Returns
    -------
    str
        The resolved tracking URI (also written to the environment).
    """
    # ── 1. Respect user-supplied value ───────────────────────────────────────
    existing = os.environ.get("MLFLOW_TRACKING_URI", "")
    if existing:
        uri = existing
    # ── 2. Legacy file-store auto-detect ─────────────────────────────────────
    elif _has_legacy_mlruns_data(workspace_root):
        mlruns_abs = os.path.abspath(os.path.join(workspace_root, "mlruns"))
        uri = f"file://{mlruns_abs}"
        print(
            f"\n⚠️  [QuantPits] Legacy MLflow file-store detected at:\n"
            f"     {mlruns_abs}\n"
            f"   Falling back to file:// backend (MLFLOW_ALLOW_FILE_STORE=true).\n"
            f"   To migrate to the modern SQLite backend run:\n"
            f"     python -m quantpits.tools.migrate_mlflow_backend "
            f"--workspace {workspace_root}\n",
            flush=True,
        )
    # ── 3. New default: SQLite ────────────────────────────────────────────────
    else:
        db_path = os.path.abspath(os.path.join(workspace_root, "mlflow.db"))
        uri = f"sqlite:///{db_path}"

    os.environ["MLFLOW_TRACKING_URI"] = uri

    # Ensure file:// backend never blocks on mlflow ≥ 3.0
    if uri.startswith("file://"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    return uri


# 确保 MLflow 的实验数据也隔离存放在当前 Workspace 下
# Backend resolution: sqlite:// (new default) or file:// (legacy/user-override).
mlflow_backend = _resolve_mlflow_backend(ROOT_DIR)

# mlruns_dir kept for backward-compat; None when using the SQLite backend.
mlruns_dir = (
    os.path.abspath(os.path.join(ROOT_DIR, "mlruns"))
    if mlflow_backend.startswith("file://")
    else None
)

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


def set_root_dir(path: str):
    """运行时切换工作区根目录（供 Playground / 多 workspace 场景使用）

    Updates:
    - env.ROOT_DIR, env.mlruns_dir, env.mlflow_backend
    - os.environ QLIB_WORKSPACE_DIR and MLFLOW_TRACKING_URI
    - train_utils module-level path constants (they are value copies, not references)

    Args:
        path: Absolute path to the new workspace root directory.
    """
    global ROOT_DIR, mlruns_dir, mlflow_backend
    ROOT_DIR = os.path.abspath(path)
    os.environ["QLIB_WORKSPACE_DIR"] = ROOT_DIR

    # Clear the old URI so _resolve_mlflow_backend re-evaluates the new workspace
    # instead of treating the old value as a user-supplied override.
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    mlflow_backend = _resolve_mlflow_backend(ROOT_DIR)
    mlruns_dir = (
        os.path.abspath(os.path.join(ROOT_DIR, "mlruns"))
        if mlflow_backend.startswith("file://")
        else None
    )

    # Synchronize train_utils module-level path constants.
    # train_utils.ROOT_DIR = env.ROOT_DIR is a value copy (strings are immutable),
    # so we must explicitly update all derived paths in that module.
    import sys
    tu = sys.modules.get("quantpits.utils.train_utils")
    if tu is not None:
        tu.ROOT_DIR = ROOT_DIR
        tu.REGISTRY_FILE = os.path.join(ROOT_DIR, "config", "model_registry.yaml")
        tu.MODEL_CONFIG_FILE = os.path.join(ROOT_DIR, "config", "model_config.json")
        tu.PROD_CONFIG_FILE = os.path.join(ROOT_DIR, "config", "prod_config.json")
        tu.RECORD_OUTPUT_FILE = os.path.join(ROOT_DIR, "latest_train_records.json")
        tu.PREDICTION_OUTPUT_DIR = os.path.join(ROOT_DIR, "output", "predictions")
        tu.ROLLING_PREDICTION_DIR = os.path.join(ROOT_DIR, "output", "predictions", "rolling")
        tu.HISTORY_DIR = os.path.join(ROOT_DIR, "data", "history")
        tu.RUN_STATE_FILE = os.path.join(ROOT_DIR, "data", "run_state.json")
        tu.ROLLING_STATE_FILE = os.path.join(ROOT_DIR, "data", "rolling_state.json")
        tu.LEGACY_ROLLING_RECORD_FILE = os.path.join(ROOT_DIR, "latest_rolling_records.json")
        tu.PRETRAINED_DIR = os.path.join(ROOT_DIR, "data", "pretrained")
