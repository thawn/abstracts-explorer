#!/bin/bash
# CI Environment Setup Script for Abstracts Explorer
# This script sets up the environment for running tests in CI

set -e  # Exit on error

echo "========================================="
echo "Setting up CI environment..."
echo "========================================="

# Function to print section headers
print_section() {
    echo ""
    echo "========================================="
    echo "$1"
    echo "========================================="
}

# Parse command-line arguments
INSTALL_NODE="true"
INSTALL_PYTHON="true"
EXTRA_DEPS="--extra dev --extra web"

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-node)
            INSTALL_NODE="false"
            shift
            ;;
        --skip-python)
            INSTALL_PYTHON="false"
            shift
            ;;
        --extras)
            EXTRA_DEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if Python is available
if [ "$INSTALL_PYTHON" = "true" ]; then
    print_section "Checking Python installation"
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        echo "Error: Python 3 is not installed"
        exit 1
    fi
    
    if command -v python3 &> /dev/null; then
        python3 --version
    else
        python --version
    fi

    # Check if uv is available (should be installed by workflow)
    print_section "Checking uv installation"
    if ! command -v uv &> /dev/null; then
        echo "Error: uv is not installed"
        echo "Please install uv first: https://docs.astral.sh/uv/"
        exit 1
    fi
    uv --version

    # Install Python dependencies
    print_section "Installing Python dependencies"
    # Install with specified extras for testing
    uv sync $EXTRA_DEPS
    echo "✓ Python dependencies installed"
fi

# Check if Node.js is available for JavaScript tests
if [ "$INSTALL_NODE" = "true" ]; then
    print_section "Checking Node.js installation"
    if command -v node &> /dev/null; then
        node --version
        npm --version
        
        # Install Node.js dependencies
        print_section "Installing Node.js dependencies"
        npm ci
        echo "✓ Node.js dependencies installed"
    else
        echo "Warning: Node.js is not installed, skipping JavaScript setup"
    fi
fi

# Create necessary directories
print_section "Creating required directories"
mkdir -p data
mkdir -p chroma_db
echo "✓ Directories created"

# Verify installation
print_section "Verifying installation"
if [ -d ".venv" ]; then
    echo "✓ Virtual environment created"
fi

if [ -f "pyproject.toml" ]; then
    echo "✓ Project configuration found"
fi

if [ -d "tests" ]; then
    echo "✓ Test directory found"
fi

print_section "CI environment setup complete!"
echo "You can now run tests with: uv run pytest"
