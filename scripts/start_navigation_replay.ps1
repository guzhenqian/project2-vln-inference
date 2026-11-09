param(
    [string]$Run = 'logs/learning_baseline6_20261031_153519',
    [int]$Port = 8766
)
$projectRoot = Split-Path $PSScriptRoot -Parent
if (-not [System.IO.Path]::IsPathRooted($Run)) { $Run = Join-Path $projectRoot $Run }
python (Join-Path $projectRoot 'checks/serve_navigation_replay.py') --run $Run --port $Port
