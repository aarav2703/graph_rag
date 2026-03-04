$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = "$repoRoot\src"
Write-Host "PYTHONPATH set to $env:PYTHONPATH"
python -c "import pmc_graphrag; print('pmc_graphrag import ok')"
