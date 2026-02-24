#!/bin/bash
# Source this file to activate the workspace
export QLIB_WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Workspace activated: $QLIB_WORKSPACE_DIR"
