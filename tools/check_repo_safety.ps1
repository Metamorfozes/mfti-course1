$ErrorActionPreference = "Stop"

$patterns = @(
  'C:\\Users\\',
  'H:\\',
  'MaxBo',
  'OPENROUTER',
  'API_KEY',
  'token',
  'sk-'
)

$found = $false
foreach ($pattern in $patterns) {
  $result = git grep -n -I -- $pattern
  if ($LASTEXITCODE -eq 0 -and $result) {
    Write-Host "FOUND pattern: $pattern"
    $result | ForEach-Object { Write-Host $_ }
    $found = $true
  }
}

if ($found) {
  exit 1
}

Write-Host "OK: no blocked patterns found."
exit 0
