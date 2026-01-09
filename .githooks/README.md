# Version-Controlled Git Hooks

This directory contains Git hooks that are tracked in version control.

## About

Git hooks in the `.git/hooks/` directory are not version controlled by default. To share hooks with the team and ensure everyone has the same setup, we store the actual hook scripts here in `.githooks/` and install them via npm scripts.

## Hooks

### pre-commit

Runs before each commit. If you're committing changes to HTML, JS, or CSS files, this hook:
1. Runs `npm run install:vendor` to rebuild vendor files
2. Checks if vendor files were modified
3. If vendor files changed, prompts you to stage and commit them

### post-checkout

Runs after checking out a branch. If HTML, JS, or CSS files changed between branches:
1. Automatically runs `npm run install:vendor`
2. Rebuilds vendor files to match the new branch

### post-merge

Runs after pulling/merging changes. If HTML, JS, or CSS files changed:
1. Automatically runs `npm run install:vendor`
2. Ensures vendor files are rebuilt after pulling changes

## Installation

Hooks are automatically installed when you run:

```bash
npm install
```

This runs the `postinstall` script which copies hooks from `.githooks/` to `.git/hooks/` and makes them executable.

## Manual Installation

To install hooks manually:

```bash
npm run setup:hooks
```

Or directly:

```bash
cp .githooks/* .git/hooks/
chmod +x .git/hooks/pre-commit .git/hooks/post-checkout .git/hooks/post-merge
```

## Updating Hooks

To update a hook:

1. Edit the script in `.githooks/`
2. Test it locally
3. Commit the changes
4. Team members will get the updated hook on their next `npm install`

## Files Monitored

The hooks monitor changes to:
- `src/abstracts_explorer/web_ui/templates/*.html`
- `src/abstracts_explorer/web_ui/static/*.js`
- `src/abstracts_explorer/web_ui/static/*.css`
- `tailwind.config.js`

## Bypassing Hooks

To temporarily bypass the pre-commit hook:

```bash
git commit --no-verify
```

To disable a specific hook:

```bash
chmod -x .git/hooks/pre-commit
```

To re-enable:

```bash
chmod +x .git/hooks/pre-commit
```

## More Information

See:
- [docs/vendor-auto-update.md](../docs/vendor-auto-update.md) - User guide
