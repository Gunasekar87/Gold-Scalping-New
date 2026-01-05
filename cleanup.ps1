# AETHER Cleanup Script
# Removes cache files and organizes workspace

Write-Host "=== AETHER Cleanup Script ===" -ForegroundColor Cyan
Write-Host ""

# 1. Remove Python cache files
Write-Host "[1/4] Removing Python cache files..." -ForegroundColor Yellow
Get-ChildItem -Path "src" -Include "__pycache__" -Recurse -Force -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path "." -Include "*.pyc" -Recurse -Force -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Write-Host "   Cache files removed" -ForegroundColor Green

# 2. Remove temporary files
Write-Host "[2/4] Removing temporary files..." -ForegroundColor Yellow
Get-ChildItem -Path "." -Include "*.tmp","*.bak","*.old","*.backup" -Recurse -Force -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Write-Host "   Temporary files removed" -ForegroundColor Green

# 3. Clean logs (keep last 7 days)
Write-Host "[3/4] Cleaning old logs..." -ForegroundColor Yellow
$cutoffDate = (Get-Date).AddDays(-7)
Get-ChildItem -Path "logs" -File -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -lt $cutoffDate } | Remove-Item -Force -ErrorAction SilentlyContinue
Write-Host "   Old logs removed" -ForegroundColor Green

# 4. Verify state files
Write-Host "[4/4] Verifying state files..." -ForegroundColor Yellow
$stateFiles = @("data\position_state.json", "data\brain_memory.json", "data\optimizer_state.json")
foreach ($file in $stateFiles) {
    if (Test-Path $file) {
        $size = (Get-Item $file).Length / 1KB
        Write-Host "   $file - $([math]::Round($size, 2)) KB" -ForegroundColor Green
    } else {
        Write-Host "   $file - Missing (will be created)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=== Cleanup Complete ===" -ForegroundColor Cyan
Write-Host "Your workspace is clean and organized!" -ForegroundColor Green
