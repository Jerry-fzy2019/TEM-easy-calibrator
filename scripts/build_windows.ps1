param(
    [switch]$IncludeDm3,
    [string]$Version = "0.1.0",
    [switch]$SkipInstaller,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $true
}

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$VenvDir = Join-Path $Root ".venv-build"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$DistDir = Join-Path $Root "dist"
$AppDir = Join-Path $DistDir "TEM Easy Calibrator"
$PortableZip = Join-Path $DistDir "TEM-Easy-Calibrator-windows-portable.zip"

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList
    )

    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($ArgumentList -join ' ')"
    }
}

function Get-PythonCommand {
    if ($Python) {
        return @($Python)
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        try {
            & py -3.11 -c "import sys; print(sys.version)" *> $null
            if ($LASTEXITCODE -eq 0) {
                return @("py", "-3.11")
            }
        }
        catch {
            # Fall back to python below.
        }
    }

    return @("python")
}

function Get-SelectedPythonPrefixArgs {
    if ($Script:SelectedPython.Count -le 1) {
        return @()
    }
    return $Script:SelectedPython[1..($Script:SelectedPython.Count - 1)]
}

function Invoke-SelectedPython {
    param([string[]]$ArgumentList)

    $cmd = $Script:SelectedPython[0]
    $prefixArgs = Get-SelectedPythonPrefixArgs
    Invoke-Checked $cmd ($prefixArgs + $ArgumentList)
}

function Get-PythonVersionText {
    param([string]$PythonExe)

    & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($LASTEXITCODE -ne 0) {
        return ""
    }
}

$Script:SelectedPython = @(Get-PythonCommand)
$SelectedPythonCommand = $Script:SelectedPython[0]
$SelectedVersion = & $SelectedPythonCommand @(Get-SelectedPythonPrefixArgs) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ([version]$SelectedVersion -lt [version]"3.10" -or [version]$SelectedVersion -ge [version]"3.13") {
    throw "Please build with Python 3.10, 3.11, or 3.12. Current Python is $SelectedVersion. Install Python 3.11 or pass -Python C:\Path\To\python.exe."
}

if (Test-Path $PythonExe) {
    $VenvVersion = Get-PythonVersionText $PythonExe
    if (-not $VenvVersion -or [version]$VenvVersion -lt [version]"3.10" -or [version]$VenvVersion -ge [version]"3.13") {
        Remove-Item -LiteralPath $VenvDir -Recurse -Force
    }
}

if (-not (Test-Path $PythonExe)) {
    Invoke-SelectedPython @("-m", "venv", $VenvDir)
}

Invoke-Checked $PythonExe @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked $PythonExe @("-m", "pip", "install", "-r", "requirements.txt", "-r", "requirements-build.txt")

if ($IncludeDm3) {
    Invoke-Checked $PythonExe @("-m", "pip", "install", "-r", "requirements-dm3.txt")
    $env:INCLUDE_DM3 = "1"
}
else {
    $env:INCLUDE_DM3 = "0"
}

Invoke-Checked $PythonExe @("-m", "PyInstaller", "--noconfirm", "--clean", "packaging\pyinstaller\TEMEasyCalibrator.spec")

if (Test-Path $PortableZip) {
    Remove-Item -LiteralPath $PortableZip -Force
}
if (-not (Test-Path $AppDir)) {
    throw "PyInstaller output was not found: $AppDir"
}
Compress-Archive -Path (Join-Path $AppDir "*") -DestinationPath $PortableZip -Force

if (-not $SkipInstaller) {
    $Iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
    if (-not (Test-Path $Iscc)) {
        Write-Warning "Inno Setup was not found. Portable zip was created, installer was skipped."
        Write-Warning "Install Inno Setup 6 to build the setup exe: https://jrsoftware.org/isinfo.php"
    }
    else {
        $env:APP_VERSION = $Version
        Invoke-Checked $Iscc @("packaging\windows\installer.iss")
    }
}

Write-Host "Build artifacts:"
Get-ChildItem -LiteralPath $DistDir -File | Select-Object -ExpandProperty FullName
