import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 优先使用环境变量 QLIB_WORKSPACE_DIR
if "QLIB_WORKSPACE_DIR" in os.environ:
    ROOT_DIR = os.path.abspath(os.environ["QLIB_WORKSPACE_DIR"])
else:
    # 2. 回退到传统的硬编码相对路径 (向后兼容)
    # 如果 env.py 位于 engine/scripts 目录下，ROOT_DIR 应该是 QuantPits 根目录 (即 SCRIPT_DIR 的上两级)
    ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# 确保 MLflow 的实验数据 (mlruns) 也隔离存放在当前 Workspace 下
# 使用绝对路径确保无论从哪里执行，路径都指向正确位置
mlruns_dir = os.path.abspath(os.path.join(ROOT_DIR, 'mlruns'))
os.environ["MLFLOW_TRACKING_URI"] = f"file://{mlruns_dir}"
