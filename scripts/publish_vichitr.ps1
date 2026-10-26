param(
    [Parameter(Mandatory=$true)][string]$Name,
    [Parameter(Mandatory=$true)][string]$Email,
    [string]$Destination = (Join-Path $HOME 'CheckpointRL-vichitr-balanced'),
    [string]$Source = '',
    [string]$Remote = 'https://github.com/vichitrarora/CheckpointRL',
    [string]$Python = 'python',
    [switch]$Once
)
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($Source)) {
    $Source = Split-Path -Parent $PSScriptRoot
}
$publishArgs = @((Join-Path $PSScriptRoot 'publish_history.py'), '--owner', 'vichitr', '--name', $Name, '--email', $Email, '--source', $Source, '--destination', $Destination, '--remote', $Remote)
if ($Once) { $publishArgs += '--once' }
& $Python @publishArgs
if ($LASTEXITCODE -ne 0) { throw "Publication stopped (exit $LASTEXITCODE). See the message above." }
