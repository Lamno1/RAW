$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $workspace
$logDirectory = Join-Path $workspace 'runs\s8_multiseed\orchestration'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$python = (Get-Command python -ErrorAction Stop).Source
$statusPath = Join-Path $logDirectory 'status.json'

function Write-RunStatus {
    param([string]$State, [int]$Seed, [string]$Phase, [int]$ExitCode)
    $record = [ordered]@{
        state = $State
        seed = $Seed
        phase = $Phase
        exit_code = $ExitCode
        updated_utc = (Get-Date).ToUniversalTime().ToString('o')
        test_split_access = $false
    }
    $temporary = "$statusPath.tmp"
    $record | ConvertTo-Json | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}

foreach ($seed in @(42, 123, 2026)) {
    foreach ($phase in @('training', 'validation')) {
        if ($phase -eq 'training') {
            $script = 'train_camo_s8_multiseed.py'
        } else {
            $script = 'evaluate_camo_s8_multiseed_full_cloud.py'
        }
        $prefix = Join-Path $logDirectory "${phase}_seed${seed}"
        if (Test-Path -LiteralPath "$prefix.launch_error.log") {
            $prefix = "${prefix}_attempt002"
        }
        Write-RunStatus -State 'RUNNING' -Seed $seed -Phase $phase -ExitCode 0
        try {
            $previousPreference = $ErrorActionPreference
            $ErrorActionPreference = 'Continue'
            & $python $script --seed $seed 1> "$prefix.stdout.log" 2> "$prefix.stderr.log"
            $exitCode = $LASTEXITCODE
            $ErrorActionPreference = $previousPreference
            if ($exitCode -ne 0) {
                Write-RunStatus -State 'FAILED' -Seed $seed -Phase $phase -ExitCode $exitCode
                exit $exitCode
            }
        } catch {
            $_ | Out-String | Set-Content -LiteralPath "$prefix.launch_error.log" -Encoding utf8
            Write-RunStatus -State 'FAILED_TO_LAUNCH' -Seed $seed -Phase $phase -ExitCode 1
            exit 1
        }
    }
}
Write-RunStatus -State 'COMPLETE' -Seed 2026 -Phase 'validation' -ExitCode 0
