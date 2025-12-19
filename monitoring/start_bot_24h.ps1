param(
  [int]$Hours = 24,
  [int]$ScanIntervalSeconds = 300
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $root

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
  throw "Python venv not found at $python. Activate venv or create it first."
}

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$out = Join-Path $root ("logs\run_{0}.out" -f $ts)
$err = Join-Path $root ("logs\run_{0}.err" -f $ts)
$health = Join-Path $root ("logs\health_{0}.txt" -f $ts)
$pidFile = Join-Path $root 'logs\run_bot.pid'

"=== Aether 24h Run ===" | Out-File -FilePath $out -Encoding utf8

"=== Aether 24h Health ===" | Out-File -FilePath $health -Encoding utf8
("Started: {0}" -f (Get-Date).ToString('s')) | Add-Content -Path $health
("Out: {0}" -f $out) | Add-Content -Path $health
("Err: {0}" -f $err) | Add-Content -Path $health

# Launch bot detached, with robust stdio redirects.
$proc = Start-Process -FilePath $python -ArgumentList @('run_bot.py') -WorkingDirectory $root -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
$proc.Id | Set-Content -Path $pidFile

("PID: {0}" -f $proc.Id) | Add-Content -Path $health
("Health log: {0}" -f $health) | Add-Content -Path $health

$deadline = (Get-Date).AddHours($Hours)

# Simple periodic scan (counts only) to avoid huge duplication.
while ((Get-Date) -lt $deadline) {
  Start-Sleep -Seconds $ScanIntervalSeconds

  if ($proc.HasExited) {
    ("[{0}] BOT EXITED early. ExitCode={1}" -f (Get-Date).ToString('s'), $proc.ExitCode) | Add-Content -Path $health
    break
  }

  $patterns = @('Traceback', 'CRITICAL', 'ERROR', 'Exception', 'retcode=', 'absence of network connection')
  $tailN = 400

  $errTail = ''
  if (Test-Path $err) {
    $errTail = (Get-Content -Path $err -Tail $tailN -ErrorAction SilentlyContinue | Out-String)
  }

  $counts = @{}
  foreach ($p in $patterns) {
    $counts[$p] = ([regex]::Matches($errTail, [regex]::Escape($p))).Count
  }

  $line = "[{0}] alive pid={1} | tail({2}) err matches: {3}" -f (Get-Date).ToString('s'), $proc.Id, $tailN, (($counts.GetEnumerator() | Sort-Object Name | ForEach-Object { "{0}={1}" -f $_.Name, $_.Value }) -join ', ')
  $line | Add-Content -Path $health
}

# Stop after duration, if still alive.
if (-not $proc.HasExited) {
  ("[{0}] STOPPING after {1}h" -f (Get-Date).ToString('s'), $Hours) | Add-Content -Path $health
  try {
    Stop-Process -Id $proc.Id -Force -ErrorAction Stop
  } catch {
    ("[{0}] Stop-Process failed: {1}" -f (Get-Date).ToString('s'), $_.Exception.Message) | Add-Content -Path $health
  }
}

("[{0}] DONE" -f (Get-Date).ToString('s')) | Add-Content -Path $health
