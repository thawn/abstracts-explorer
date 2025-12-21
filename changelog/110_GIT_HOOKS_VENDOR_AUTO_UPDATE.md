# Git Hooks for Automatic Vendor File Updates

## Summary

Added Git hooks that automatically run `npm run install:vendor` when HTML, JavaScript, or CSS files change. This ensures vendor files (Font Awesome, Marked.js, KaTeX, and Tailwind CSS) stay synchronized with source code changes.

## Changes

### Git Hooks Created

Hooks are stored in `.githooks/` (version controlled) and installed to `.git/hooks/` via npm scripts:

1. **pre-commit hook** (`.githooks/pre-commit`)
   - Runs before each commit
   - Detects staged changes to HTML, JS, CSS files
   - Rebuilds vendor files automatically
   - Prompts to stage vendor changes if they were modified
   - Prevents commits with out-of-sync vendor files

2. **post-checkout hook** (`.githooks/post-checkout`)
   - Runs after checking out branches
   - Detects file changes between branches
   - Automatically rebuilds vendor files after branch switch

3. **post-merge hook** (`.githooks/post-merge`)
   - Runs after pulling or merging changes
   - Detects changes in merged code
   - Automatically rebuilds vendor files after merge

**Version Control**: Hooks are stored in `.githooks/` directory (tracked by Git) and copied to `.git/hooks/` during installation. This ensures all team members have the same hook configuration.

### Files Monitored

The hooks monitor changes to:
- `src/neurips_abstracts/web_ui/templates/*.html`
- `src/neurips_abstracts/web_ui/static/*.js`
- `src/neurips_abstracts/web_ui/static/*.css`
- `tailwind.config.js`

### package.json Updates

Added new npm scripts:
- `postinstall`: Automatically runs after `npm install` to set up hooks and vendor files
- `setup:hooks`: Makes Git hooks executable (cross-platform compatible)

### Documentation

Created `.git/hooks/README.md` with:
- Explanation of each hook's purpose
- Setup and troubleshooting instructions
- How to temporarily disable hooks
- What `npm run install:vendor` does

## Impact

### For Developers

- **No manual intervention**: Vendor files automatically update when source changes
- **Prevents mistakes**: Can't commit changes without rebuilding vendor files
- **Smooth branch switching**: Vendor files stay in sync when switching branches
- **After pulling**: Vendor files automatically update after pulling changes

### Workflow Example

```bash
# Edit a JavaScript file
vim src/neurips_abstracts/web_ui/static/app.js

# Try to commit
git add src/neurips_abstracts/web_ui/static/app.js
git commit -m "Update search feature"

# Hook automatically:
# 1. Detects JS file change
# 2. Runs npm run install:vendor
# 3. Rebuilds Tailwind CSS and other vendor files
# 4. Prompts to stage vendor changes if needed
```

### Setup

Hooks are automatically installed when running:
```bash
npm install
```

The `postinstall` script:
1. Runs `npm run install:vendor` to build all vendor files
2. Runs `npm run setup:hooks` to make hooks executable

### Manual Setup

If needed, hooks can be set up manually:
```bash
chmod +x .git/hooks/post-checkout .git/hooks/post-merge .git/hooks/pre-commit
```

## Testing

Verified that:
- All hooks are executable (`-rwxr-xr-x` permissions)
- Hooks are in correct location (`.git/hooks/`)
- Scripts in `package.json` are properly configured
- Documentation is comprehensive

## Technical Details

### Hook Execution

**pre-commit**:
- Uses `git diff --cached` to detect staged files
- Runs `npm run install:vendor` if matches found
- Checks for modified vendor files after build
- Exits with code 1 if vendor files need staging

**post-checkout**:
- Receives previous HEAD, new HEAD, and checkout type as arguments
- Only runs on branch checkouts (not file checkouts)
- Uses `git diff --name-only` to compare HEAD refs
- Runs vendor build if monitored files changed

**post-merge**:
- Uses `git merge-base HEAD@{1} HEAD` to find merge base
- Compares merge base with current HEAD
- Runs vendor build if monitored files changed

### Why These Hooks?

- **pre-commit**: Prevents committing source without vendor updates
- **post-checkout**: Handles branch switching scenarios
- **post-merge**: Handles pull/merge scenarios

Together, these cover all common Git workflows where vendor files might become out of sync.

## Benefits

1. **Consistency**: Vendor files always match source code
2. **Developer Experience**: No need to remember to rebuild
3. **Error Prevention**: Can't accidentally commit stale vendor files
4. **Team Collaboration**: Everyone's environment stays consistent
5. **CI/CD Ready**: Hooks work in automated environments

## Disabling Hooks (if needed)

To bypass pre-commit hook temporarily:
```bash
git commit --no-verify
```

To disable a hook:
```bash
chmod -x .git/hooks/pre-commit
```

To re-enable:
```bash
chmod +x .git/hooks/pre-commit
```

## Related Files

- `.githooks/pre-commit` - Pre-commit hook script (version controlled)
- `.githooks/post-checkout` - Post-checkout hook script (version controlled)
- `.githooks/post-merge` - Post-merge hook script (version controlled)
- `.githooks/README.md` - Hook documentation and maintenance guide
- `.git/hooks/*` - Installed hooks (not version controlled, auto-generated)
- `package.json` - Updated with postinstall and setup:hooks scripts
- `docs/vendor-auto-update.md` - User-facing documentation

## Future Enhancements

Possible improvements:
- Add Git hook for `post-rewrite` (handles rebase, amend)
- Make hook paths configurable
- Add option to skip vendor rebuild with environment variable
- Create a `validate:vendor` script to check if rebuild is needed
