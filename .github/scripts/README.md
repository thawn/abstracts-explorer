# CI Setup Scripts

This directory contains scripts for setting up the CI environment for the NeurIPS Abstracts project.

## Scripts

### setup-ci.sh

Bash script for Unix/macOS environments that sets up Python, uv, and Node.js dependencies.

**Usage:**

```bash
# Basic usage (installs dev and web extras)
.github/scripts/setup-ci.sh

# Skip Node.js installation
.github/scripts/setup-ci.sh --skip-node

# Custom extras
.github/scripts/setup-ci.sh --extras "--extra docs"

# Multiple flags
.github/scripts/setup-ci.sh --skip-node --extras "--extra dev"
```

**Options:**

- `--skip-node`: Skip Node.js dependency installation
- `--skip-python`: Skip Python dependency installation
- `--extras <string>`: Specify custom extras to install (default: "--extra dev --extra web")

**Requirements:**

- Python 3.11 or higher
- uv package manager
- Node.js and npm (if not using --skip-node)

### setup-ci.ps1

PowerShell script for Windows environments with the same functionality as setup-ci.sh.

**Usage:**

```powershell
# Basic usage
.github/scripts/setup-ci.ps1

# Skip Node.js installation
.github/scripts/setup-ci.ps1 --skip-node

# Custom extras
.github/scripts/setup-ci.ps1 --extras "--extra docs"
```

**Options:**

Same as setup-ci.sh

## What the Scripts Do

1. **Verify Prerequisites**: Check that Python and uv are installed
2. **Install Python Dependencies**: Run `uv sync` with specified extras
3. **Install Node.js Dependencies**: Run `npm ci` (if Node.js is available and not skipped)
4. **Create Directories**: Create required data and chroma_db directories
5. **Verify Installation**: Check that virtual environment and project files exist

## Exit Codes

- `0`: Success
- `1`: Error (missing prerequisites, invalid arguments, or installation failure)

## Environment Variables

The scripts use the following environment variables (set automatically by GitHub Actions):

- `GITHUB_WORKSPACE`: The workspace directory
- `RUNNER_OS`: The operating system (Linux, macOS, Windows)

## Integration with GitHub Actions

These scripts are called by:

1. **Composite Action**: `.github/actions/setup-environment/action.yml`
2. **Workflow Files**: Directly from `.github/workflows/tests.yml`

See the composite action documentation for the recommended way to use these scripts in workflows.

## Local Development

You can run these scripts locally to set up your development environment:

```bash
# Clone the repository
git clone https://github.com/thawn/neurips-abstracts.git
cd neurips-abstracts

# Install uv first (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run the setup script
.github/scripts/setup-ci.sh

# Or with custom options
.github/scripts/setup-ci.sh --skip-node --extras "--extra docs"
```

## Troubleshooting

**Error: uv is not installed**
- Install uv: https://docs.astral.sh/uv/

**Error: Python 3 is not installed**
- Install Python 3.11 or higher

**Warning: Node.js is not installed**
- This is only a warning if `--skip-node` is not used
- Install Node.js 20+ from https://nodejs.org/

## Contributing

When modifying these scripts:

1. Test on all platforms (Linux, macOS, Windows)
2. Update this README with any new options or behavior
3. Validate bash syntax: `bash -n setup-ci.sh`
4. Update the changelog entry
5. Update the composite action if the script interface changes
