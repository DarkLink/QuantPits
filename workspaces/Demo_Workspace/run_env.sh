#!/bin/bash
# Source this file to activate the workspace
export QLIB_WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Uncomment and modify to use a custom Qlib data directory:
# export QLIB_DATA_DIR="~/.qlib/qlib_data/cn_data"
# export QLIB_REGION="cn"
echo "Workspace activated: $QLIB_WORKSPACE_DIR"
