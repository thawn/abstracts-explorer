# CI Environment Setup Script for NeurIPS Abstracts (Windows)
# This script sets up the environment for running tests in CI on Windows

$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Setting up CI environment (Windows)..." -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

function Print-Section {
    param([string]$message)
    Write-Host ""
    Write-Host "=========================================" -ForegroundColor Cyan
    Write-Host $message -ForegroundColor Cyan
    Write-Host "=========================================" -ForegroundColor Cyan
}

# Parse command-line arguments
$InstallNode = $true
$InstallPython = $true

foreach ($arg in $args) {
    switch ($arg) {
        "--skip-node" { $InstallNode = $false }
        "--skip-python" { $InstallPython = $false }
        default {
            Write-Host "Unknown option: $arg" -ForegroundColor Red
            exit 1
        }
    }
}

# Check if Python is available
if ($InstallPython) {
    Print-Section "Checking Python installation"
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        Write-Host "Error: Python is not installed" -ForegroundColor Red
        exit 1
    }
    python --version

    # Check if uv is available (should be installed by workflow)
    Print-Section "Checking uv installation"
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uvCmd) {
        Write-Host "Error: uv is not installed" -ForegroundColor Red
        Write-Host "Please install uv first: https://docs.astral.sh/uv/" -ForegroundColor Yellow
        exit 1
    }
    uv --version

    # Install Python dependencies
    Print-Section "Installing Python dependencies"
    uv sync --extra dev --extra web
    Write-Host "✓ Python dependencies installed" -ForegroundColor Green
}

# Check if Node.js is available for JavaScript tests
if ($InstallNode) {
    Print-Section "Checking Node.js installation"
    $nodeCmd = Get-Command node -ErrorAction SilentlyContinue
    if ($nodeCmd) {
        node --version
        npm --version
        
        # Install Node.js dependencies
        Print-Section "Installing Node.js dependencies"
        npm ci
        Write-Host "✓ Node.js dependencies installed" -ForegroundColor Green
    }
    else {
        Write-Host "Warning: Node.js is not installed, skipping JavaScript setup" -ForegroundColor Yellow
    }
}

# Create necessary directories
Print-Section "Creating required directories"
New-Item -ItemType Directory -Force -Path "data" | Out-Null
New-Item -ItemType Directory -Force -Path "chroma_db" | Out-Null
Write-Host "✓ Directories created" -ForegroundColor Green

# Verify installation
Print-Section "Verifying installation"
if (Test-Path ".venv") {
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

if (Test-Path "pyproject.toml") {
    Write-Host "✓ Project configuration found" -ForegroundColor Green
}

if (Test-Path "tests") {
    Write-Host "✓ Test directory found" -ForegroundColor Green
}

Print-Section "CI environment setup complete!"
Write-Host "You can now run tests with: uv run pytest" -ForegroundColor Green
