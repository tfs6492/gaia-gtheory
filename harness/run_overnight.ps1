<#
run_overnight.ps1
=================
Drives the full GAIA data collection (the fully crossed design):
  * 53 tasks x 2 efforts x 2 scaffolds x 20 runs = 4240 independent agent runs
    (model calls go to the OpenAI API; nothing runs on the local GPU)
  * prevents the machine from sleeping for the duration of the run only
    (normal power behaviour is restored automatically when it finishes)
  * writes fresh, timestamped output so nothing from a previous run carries over:
      raw/          -> full_*.jsonl (source of truth), runs_txt_*/ , run_console_*.txt
      data/private/ -> gaia_runs_*.csv and gaia_runs_*_summary.csv (contain GAIA answers)
  * builds the analysis CSVs automatically at the end

Usage (from the repo root):
    python harness/fetch_gaia.py          # once: gated GAIA download -> data/private/
    $env:OPENAI_API_KEY = "sk-..."        # for gpt-* models
    .\harness\run_overnight.ps1

Optional overrides:
    .\harness\run_overnight.ps1 -Limit 2 -Runs 2               # quick dry run
    .\harness\run_overnight.ps1 -Shuffle -ExcludeAttachments   # recommended for new runs
    .\harness\run_overnight.ps1 -Efforts "low,high" -Scaffolds "tool,code" -Search serper
#>

param(
    [string]$Model        = "gpt-5.4-mini",
    [int]   $Runs         = 20,
    [string]$Tasks        = "data/private/gaia_level1.json",
    [string]$Efforts      = "low,high",   # crossed facet: reasoning_effort levels
    [string]$Scaffolds    = "tool,code",  # crossed facet: manager scaffold (tool|code)
    [string]$Search       = "serper",     # web-search backend: serper | tavily | ddg
    [int]   $Limit        = 0,            # >0 = only the first N tasks (for a quick dry run)
    [switch]$Shuffle,                     # randomize run order (time not confounded with condition)
    [switch]$ExcludeAttachments           # drop tasks whose attachments are never supplied
)

$ErrorActionPreference = "Stop"
# This script lives in harness/; every path below is relative to the repo root.
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -Path $RepoRoot

# --- API keys -------------------------------------------------------------- #
# Fall back to the persistent User environment for any key not set in-session.
foreach ($k in "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "SERPER_API_KEY", "TAVILY_API_KEY") {
    if (-not (Get-Item "env:$k" -ErrorAction SilentlyContinue)) {
        $v = [Environment]::GetEnvironmentVariable($k, 'User')
        if ($v) { Set-Item "env:$k" $v }
    }
}
if ($Model -like "gpt*" -and -not $env:OPENAI_API_KEY) {
    Write-Error "OPENAI_API_KEY is not set. Run:  `$env:OPENAI_API_KEY = 'sk-...'  then re-launch."
    exit 1
}
if (-not (Test-Path $Tasks)) {
    Write-Error "Task file '$Tasks' not found. Run:  python harness/fetch_gaia.py"
    exit 1
}
# Only the selected search backend's key is required.
switch ($Search) {
    "serper" { if (-not $env:SERPER_API_KEY) { Write-Error "SERPER_API_KEY is not set (needed for -Search serper)."; exit 1 } }
    "tavily" { if (-not $env:TAVILY_API_KEY) { Write-Error "TAVILY_API_KEY is not set (needed for -Search tavily)."; exit 1 } }
    "ddg"    { }   # no key required
    default  { Write-Error "Unknown -Search '$Search' (expected: serper | tavily | ddg)."; exit 1 }
}

# --- Fresh, timestamped outputs (no appending to old data) ------------------ #
New-Item -ItemType Directory -Force -Path "data/private", "raw" | Out-Null
$stamp   = Get-Date -Format "yyyyMMdd_HHmmss"
$safeMod = ($Model -replace '[^A-Za-z0-9]', '_')
$jsonl   = "raw/full_${safeMod}_${stamp}.jsonl"
$txtDir  = "raw/runs_txt_${safeMod}_${stamp}"
$csv     = "data/private/gaia_runs_${safeMod}_${stamp}.csv"
$console = "raw/run_console_${safeMod}_${stamp}.txt"

$nEff  = ($Efforts   -split ',').Where({ $_.Trim() -ne '' }).Count
$nScaf = ($Scaffolds -split ',').Where({ $_.Trim() -ne '' }).Count

Write-Host "=== GAIA overnight run ==="
Write-Host "model      : $Model"
Write-Host "design     : tasks x $nEff efforts ($Efforts) x $nScaf scaffolds ($Scaffolds) x $Runs runs"
Write-Host "order      : $(if ($Shuffle) { 'shuffled' } else { 'nested (task > effort > scaffold > run)' })"
Write-Host "search     : $Search"
Write-Host "jsonl out  : $jsonl"
Write-Host "csv out    : $csv"
Write-Host "console log: $console"
Write-Host "=========================="

# --- Keep the system awake for the duration of THIS script only ------------ #
Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
$ES_CONTINUOUS      = [uint32]2147483648  # 0x80000000
$ES_SYSTEM_REQUIRED = [uint32]1           # 0x00000001
[void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)

$python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
try {
    # The harness prints progress to stderr. Under 'Stop', PowerShell 5.1 turns
    # each native stderr line into a terminating error, so use 'Continue' and
    # rely on $LASTEXITCODE instead.
    $ErrorActionPreference = "Continue"
    $pyArgs = @(
        "harness/gaia_gt_pilot.py",
        "--tasks", $Tasks, "--model", $Model, "--runs", $Runs,
        "--out", $jsonl, "--txt-dir", $txtDir,
        "--efforts", $Efforts, "--scaffolds", $Scaffolds, "--search", $Search
    )
    if ($Limit -gt 0)        { $pyArgs += @("--limit", $Limit) }
    if ($Shuffle)            { $pyArgs += "--shuffle" }
    if ($ExcludeAttachments) { $pyArgs += "--exclude-attachments" }
    & $python @pyArgs 2>&1 | Tee-Object -FilePath $console
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Harness exited with code $LASTEXITCODE (data up to the failure point is in $jsonl)."
    }

    Write-Host "`n=== Run complete. Aggregating to CSV... ==="
    & $python harness/aggregate_to_csv.py --in $jsonl --tasks $Tasks --out $csv
}
finally {
    [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS)
    Write-Host "`n[power] sleep prevention released."
}

Write-Host "`nDONE. Private CSVs:"
Write-Host "  $csv  (one row per run)"
Write-Host "Next: set PRIVATE_RUNS in R/00_config.R to this file and run R/00_prepare_data.R"
