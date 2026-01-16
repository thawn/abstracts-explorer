# GitHub Workflows Path Filters

This document explains the path filters applied to each GitHub workflow to optimize CI/CD performance by running only necessary workflows based on file changes.

## Overview

Path filters have been added to all workflows to skip unnecessary runs when only unrelated files change. This significantly reduces CI/CD resource usage while maintaining comprehensive testing coverage.

## Workflow Configurations

### 1. Tests Workflow (`tests.yml`)

**Purpose**: Runs Python tests (across multiple OS/Python versions) and JavaScript tests

**Triggers on changes to**:
- `src/**/*.py` - Python source code
- `tests/**/*.py` - Python test files
- `src/abstracts_explorer/web_ui/static/**/*.js` - JavaScript source code
- `src/abstracts_explorer/web_ui/templates/**` - HTML templates
- `pyproject.toml` - Python dependencies
- `uv.lock` - Python dependency lock file
- `package.json` - JavaScript dependencies
- `package-lock.json` - JavaScript dependency lock file
- `.github/workflows/tests.yml` - The workflow itself
- `.github/scripts/setup-ci.sh` - CI setup script (Unix)
- `.github/scripts/setup-ci.ps1` - CI setup script (Windows)
- `.github/actions/setup-environment/**` - Reusable setup action

**Skips when**: Only documentation, Docker files, or other unrelated files change

---

### 2. Lint Workflow (`lint.yml`)

**Purpose**: Runs Python linting (ruff) and type checking (mypy)

**Triggers on changes to**:
- `src/**/*.py` - Python source code
- `tests/**/*.py` - Python test files
- `pyproject.toml` - Linting configuration and dependencies
- `uv.lock` - Python dependency lock file
- `.github/workflows/lint.yml` - The workflow itself

**Skips when**: Only JavaScript, documentation, Docker files, or other unrelated files change

---

### 3. Documentation Workflow (`docs.yml`)

**Purpose**: Builds Sphinx documentation and deploys to GitHub Pages

**Triggers on changes to**:
- `docs/**` - Documentation source files
- `src/**/*.py` - Python source code (for API documentation)
- `pyproject.toml` - Documentation dependencies
- `uv.lock` - Python dependency lock file
- `.github/workflows/docs.yml` - The workflow itself
- `.github/actions/setup-environment/**` - Reusable setup action

**Skips when**: Only JavaScript tests, Docker files, or other unrelated files change

---

### 4. Docker Workflow (`docker.yml`)

**Purpose**: Builds and pushes Docker images to registries

**Triggers on changes to**:
- `Dockerfile` - Docker build instructions
- `docker-compose.yml` - Docker composition file
- `.dockerignore` - Docker ignore patterns
- `src/**` - All source code (Python, JavaScript, templates)
- `pyproject.toml` - Python dependencies
- `uv.lock` - Python dependency lock file
- `package.json` - JavaScript dependencies
- `package-lock.json` - JavaScript dependency lock file
- `tailwind.config.js` - CSS framework configuration
- `.github/workflows/docker.yml` - The workflow itself

**Skips when**: Only documentation or test files change

**Note**: This workflow also triggers on tags (`v*.*.*`) and always runs for tag pushes regardless of paths.

---

### 5. Slow Tests Workflow (`slow-tests.yml`)

**Purpose**: Runs slow/E2E tests with external service integration

**Triggers on changes to**:
- `src/**/*.py` - Python source code
- `tests/**/*.py` - Python test files
- `src/abstracts_explorer/web_ui/static/**/*.js` - JavaScript source code
- `src/abstracts_explorer/web_ui/templates/**` - HTML templates
- `pyproject.toml` - Python dependencies
- `uv.lock` - Python dependency lock file
- `package.json` - JavaScript dependencies
- `package-lock.json` - JavaScript dependency lock file
- `.github/workflows/slow-tests.yml` - The workflow itself
- `.github/scripts/setup-ci.sh` - CI setup script

**Skips when**: Only documentation or Docker files change

**Note**: This workflow also runs on a daily schedule (`cron: '0 2 * * *'`) and always runs for scheduled triggers regardless of paths.

---

## Special Considerations

### workflow_dispatch

All workflows retain the `workflow_dispatch` trigger, allowing manual runs regardless of path filters. This is useful for:
- Testing workflow changes
- Running workflows when path filters might have been too restrictive
- Emergency deployments or documentation updates

### Scheduled Runs

The `slow-tests.yml` workflow has a scheduled trigger that runs daily at 2 AM UTC. Scheduled runs ignore path filters and always execute.

### Tag Pushes

The `docker.yml` workflow triggers on version tags (`v*.*.*`). Tag pushes ignore path filters and always execute to ensure releases are properly containerized.

## Benefits

1. **Reduced CI/CD Time**: Workflows skip when irrelevant files change
2. **Resource Efficiency**: Fewer unnecessary workflow runs save compute resources
3. **Faster Feedback**: Developers get faster feedback on relevant changes
4. **Cost Savings**: Reduced GitHub Actions minutes usage

## Examples

### Example 1: Documentation-Only Change
If you only modify files in `docs/`, the following will run:
- ✅ `docs.yml` (Documentation)

The following will skip:
- ⏭️ `tests.yml` (Tests)
- ⏭️ `lint.yml` (Lint)
- ⏭️ `docker.yml` (Docker)
- ⏭️ `slow-tests.yml` (Slow Tests)

### Example 2: Python Code Change
If you modify a file in `src/abstracts_explorer/`, all workflows will run:
- ✅ `tests.yml` (Tests)
- ✅ `lint.yml` (Lint)
- ✅ `docs.yml` (Documentation - because API docs might change)
- ✅ `docker.yml` (Docker)
- ✅ `slow-tests.yml` (Slow Tests)

### Example 3: JavaScript Code Change
If you modify a file in `src/abstracts_explorer/web_ui/static/`, the following will run:
- ✅ `tests.yml` (Tests - includes JavaScript tests)
- ✅ `docker.yml` (Docker)
- ✅ `slow-tests.yml` (Slow Tests - includes E2E tests)

The following will skip:
- ⏭️ `lint.yml` (Lint - Python linting only)
- ⏭️ `docs.yml` (Documentation - JavaScript doesn't affect Python API docs)

### Example 4: README Change
If you only modify `README.md`, all workflows will skip:
- ⏭️ `tests.yml` (Tests)
- ⏭️ `lint.yml` (Lint)
- ⏭️ `docs.yml` (Documentation)
- ⏭️ `docker.yml` (Docker)
- ⏭️ `slow-tests.yml` (Slow Tests)

## Maintenance

When adding new workflows or modifying existing ones, consider:

1. **What files affect this workflow?** - Add appropriate path filters
2. **What's the impact of skipping?** - Ensure path filters aren't too restrictive
3. **Are there edge cases?** - Consider dependencies between different file types
4. **Does the workflow itself need to trigger on its own changes?** - Include the workflow file in paths

## Testing Path Filters

To test path filters locally, you can use the GitHub Actions extension in VS Code or manually check the workflow syntax:

```bash
# Validate YAML syntax
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/tests.yml'))"
```

Or use GitHub's workflow validation by pushing to a test branch and observing which workflows trigger.

---

**Last Updated**: January 16, 2026

For questions or suggestions about these path filters, please open an issue or discussion.
