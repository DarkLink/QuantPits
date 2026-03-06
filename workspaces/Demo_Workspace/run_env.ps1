# run this file to activate the workspace

$env:QLIB_WORKSPACE_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
# Uncomment and modify to use a custom Qlib data directory:
$env:QLIB_DATA_DIR = "D:\data\cn_data"
$env:QLIB_REGION = "cn"

Write-Host "Workspace activated: $env:QLIB_WORKSPACE_DIR"