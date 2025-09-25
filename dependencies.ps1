#Requires -RunAsAdministrator
# ---
# Dependency Installer for LZ5 v2.1 Converter
# ---
# This script installs all necessary components to build and run the converter:
#   1. Git for Windows (for cloning source code)
#   2. Python (as a build-time dependency and fallback)
#   3. Visual Studio 2022 Build Tools (for the C++ compiler)
#
# It uses the modern `winget` package manager for clean, automated installs.
# ---

# Stop on any error
$ErrorActionPreference = 'Stop'

# --- Helper Functions ---
function Write-Header($Title) {
    Write-Host "`n"
    Write-Host "========================================================================" -ForegroundColor Yellow
    Write-Host "  $Title" -ForegroundColor Yellow
    Write-Host "========================================================================" -ForegroundColor Yellow
}

function Test-CommandExists($CommandName) {
    return (Get-Command $CommandName -ErrorAction SilentlyContinue) -ne $null
}

# --- Main Script ---
Clear-Host
Write-Header "LZ5 v2.1 Converter - Dependency Installer"
Write-Host "This script will check for and install required development tools."
Write-Host "Administrator privileges are required."

# --- 1. Git for Windows ---
Write-Header "Step 1: Checking for Git for Windows"
if (Test-CommandExists "git") {
    Write-Host "Git is already installed." -ForegroundColor Green
    $gitPath = (Get-Command git).Source
    Write-Host "  -> Found at: $gitPath"
}
else {
    Write-Host "Git not found. Installing via winget..." -ForegroundColor Cyan
    try {
        winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
        Write-Host "Git installation complete." -ForegroundColor Green
        # Refresh environment variables to find the new command
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    }
    catch {
        Write-Host "ERROR: Git installation failed. Please install Git for Windows manually." -ForegroundColor Red
        exit 1
    }
}

# --- 2. Python ---
Write-Header "Step 2: Checking for Python"
if (Test-CommandExists "python") {
    Write-Host "Python is already installed." -ForegroundColor Green
    $pythonPath = (Get-Command python).Source
    Write-Host "  -> Found at: $pythonPath"
}
else {
    Write-Host "Python not found. Installing Python 3.11 via winget..." -ForegroundColor Cyan
    try {
        winget install --id Python.Python.3.11 -e --source winget --accept-package-agreements --accept-source-agreements
        Write-Host "Python 3.11 installation complete." -ForegroundColor Green
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    }
    catch {
        Write-Host "ERROR: Python installation failed. Please install Python 3 manually." -ForegroundColor Red
        exit 1
    }
}

# --- 3. Visual Studio 2022 Build Tools (C++ Compiler) ---
Write-Header "Step 3: Checking for Visual Studio C++ Build Tools"
if (Test-CommandExists "cl") {
    Write-Host "Visual Studio C++ Compiler (cl.exe) is already installed." -ForegroundColor Green
    $clPath = (Get-Command cl).Source
    Write-Host "  -> Found at: $clPath"
}
else {
    Write-Host "Visual Studio C++ Build Tools not found." -ForegroundColor Cyan
    Write-Host "This is a large download and may take a significant amount of time." -ForegroundColor Cyan
    Write-Host "Installing the minimal 'Desktop development with C++' workload via winget..."
    
    try {
        # This command finds the VS 2022 Build Tools package, agrees to the terms, and installs
        # only the C++ workload, which includes the compiler, linker, and essential libraries.
        winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --accept-package-agreements --accept-source-agreements --override "--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
        
        Write-Host "Visual Studio Build Tools installation complete." -ForegroundColor Green
        Write-Host "A system restart may be required for all changes to take effect." -ForegroundColor Yellow
    }
    catch {
        Write-Host "ERROR: Visual Studio Build Tools installation failed." -ForegroundColor Red
        Write-Host "Please download 'Build Tools for Visual Studio 2022' from the Microsoft website and ensure the 'Desktop development with C++' workload is selected during installation." -ForegroundColor Red
        exit 1
    }
}

# --- Final Verification ---
Write-Header "Step 4: Final Verification"
$allGood = $true

if (Test-CommandExists "git") {
    Write-Host "[OK] Git command found." -ForegroundColor Green
} else {
    Write-Host "[FAIL] Git command not found." -ForegroundColor Red
    $allGood = $false
}

if (Test-CommandExists "python") {
    Write-Host "[OK] Python command found." -ForegroundColor Green
} else {
    Write-Host "[FAIL] Python command not found." -ForegroundColor Red
    $allGood = $false
}

# The VS environment needs to be loaded, so checking for `cl` right after install
# in the same session might fail. We check for the installer helper instead.
$vswherePath = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswherePath) {
    Write-Host "[OK] Visual Studio installer found." -ForegroundColor Green
} else {
    Write-Host "[FAIL] Visual Studio installer not found." -ForegroundColor Red
    $allGood = $false
}

Write-Header "Setup Complete!"
if ($allGood) {
    Write-Host "All required dependencies are installed." -ForegroundColor Green
    Write-Host "You can now run the main 'lz5_v2.1_converter.ps1' script."
} else {
    Write-Host "Some dependencies could not be verified." -ForegroundColor Red
    Write-Host "Please review the log above. A system restart might be necessary."
}

Write-Host "Press any key to exit."
[void]$Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
