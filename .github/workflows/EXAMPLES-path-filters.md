# GitHub Workflow Path Filters - Visual Examples

This document provides visual examples showing which workflows run for different types of changes.

## Visual Matrix

| File Changed | tests.yml | lint.yml | docs.yml | docker.yml | slow-tests.yml |
|--------------|-----------|----------|----------|------------|----------------|
| `src/**/*.py` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `tests/**/*.py` | ✅ | ✅ | ❌ | ✅ | ✅ |
| `src/web_ui/static/**/*.js` | ✅ | ❌ | ❌ | ✅ | ✅ |
| `src/web_ui/templates/**` | ✅ | ❌ | ❌ | ✅ | ✅ |
| `docs/**` | ❌ | ❌ | ✅ | ❌ | ❌ |
| `README.md` | ❌ | ❌ | ❌ | ❌ | ❌ |
| `pyproject.toml` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `uv.lock` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `package.json` | ✅ | ❌ | ❌ | ✅ | ✅ |
| `Dockerfile` | ❌ | ❌ | ❌ | ✅ | ❌ |
| `.github/workflows/*.yml` | 🔀 | 🔀 | 🔀 | 🔀 | 🔀 |

**Legend:**
- ✅ = Workflow runs
- ❌ = Workflow skips
- 🔀 = Workflow runs only if it's its own file (e.g., tests.yml triggers tests.yml)

## Detailed Examples

### Example 1: Documentation Update
```bash
# Changed files:
docs/installation.md
docs/usage.md

# Workflows that run:
✅ docs.yml - Builds and deploys documentation

# Workflows that skip:
⏭️ tests.yml - No code or test changes
⏭️ lint.yml - No Python code changes
⏭️ docker.yml - No Docker or source changes
⏭️ slow-tests.yml - No code or test changes

# Time saved: ~45 minutes (9 test matrix jobs + 1 JS test + 3 other workflows)
```

### Example 2: Python Source Code Change
```bash
# Changed files:
src/abstracts_explorer/database.py
tests/test_database.py

# Workflows that run:
✅ tests.yml - Tests affected by database changes
✅ lint.yml - Python code needs linting
✅ docs.yml - API docs may have changed
✅ docker.yml - Container image needs rebuilding
✅ slow-tests.yml - Integration tests affected

# Workflows that skip:
(none)

# Time saved: 0 minutes (all workflows needed)
```

### Example 3: JavaScript Frontend Change
```bash
# Changed files:
src/abstracts_explorer/web_ui/static/modules/search.js

# Workflows that run:
✅ tests.yml - JavaScript tests need to run
✅ docker.yml - Container image needs rebuilding
✅ slow-tests.yml - E2E tests may be affected

# Workflows that skip:
⏭️ lint.yml - No Python code to lint
⏭️ docs.yml - JavaScript doesn't affect Python API docs

# Time saved: ~12 minutes (lint + docs workflows)
```

### Example 4: README Update
```bash
# Changed files:
README.md

# Workflows that run:
(none)

# Workflows that skip:
⏭️ tests.yml - README doesn't affect tests
⏭️ lint.yml - README doesn't affect linting
⏭️ docs.yml - README not in docs/ directory
⏭️ docker.yml - README doesn't affect container
⏭️ slow-tests.yml - README doesn't affect tests

# Time saved: ~60+ minutes (all workflows)
```

### Example 5: Dependency Update
```bash
# Changed files:
pyproject.toml
uv.lock

# Workflows that run:
✅ tests.yml - Dependencies affect all tests
✅ lint.yml - Linting tools may be updated
✅ docs.yml - Documentation tools may be updated
✅ docker.yml - Container dependencies changed
✅ slow-tests.yml - Dependencies affect all tests

# Workflows that skip:
(none)

# Time saved: 0 minutes (all workflows needed)
```

### Example 6: Docker Configuration Change
```bash
# Changed files:
Dockerfile
docker-compose.yml

# Workflows that run:
✅ docker.yml - Docker files directly changed

# Workflows that skip:
⏭️ tests.yml - No code changes
⏭️ lint.yml - No Python code changes
⏭️ docs.yml - No documentation changes
⏭️ slow-tests.yml - No code changes

# Time saved: ~50+ minutes (test workflows)
```

### Example 7: HTML Template Change
```bash
# Changed files:
src/abstracts_explorer/web_ui/templates/index.html

# Workflows that run:
✅ tests.yml - Templates may affect integration tests
✅ docker.yml - Container needs rebuilding
✅ slow-tests.yml - E2E tests may be affected

# Workflows that skip:
⏭️ lint.yml - No Python code to lint
⏭️ docs.yml - Templates don't affect docs

# Time saved: ~12 minutes (lint + docs workflows)
```

## Aggregate Time Savings

Based on typical development patterns:

| Change Type | Frequency | Time Saved per PR | Annual Savings* |
|-------------|-----------|-------------------|-----------------|
| Docs only | 20% | 45 min | 900 min (15 hrs) |
| README/config | 10% | 60 min | 600 min (10 hrs) |
| JavaScript only | 15% | 12 min | 180 min (3 hrs) |
| Docker only | 5% | 50 min | 250 min (4.2 hrs) |
| **Total** | **50%** | - | **1,930 min (32.2 hrs)** |

*Assumes 100 PRs per year, 50% of which benefit from path filters

## Manual Override

If you need to run all workflows regardless of paths:
```bash
# Use workflow_dispatch from GitHub UI:
1. Go to Actions tab
2. Select the workflow
3. Click "Run workflow"
4. Select branch
5. Click "Run workflow"
```

## Scheduled Runs & Tags

- **Scheduled runs** (e.g., slow-tests daily cron): Always run regardless of paths
- **Tag pushes** (e.g., v1.0.0): Docker workflow always runs regardless of paths
- **Manual dispatch**: Always runs regardless of paths

---

**Last Updated**: January 16, 2026

See also: [README-path-filters.md](./README-path-filters.md) for complete documentation.
