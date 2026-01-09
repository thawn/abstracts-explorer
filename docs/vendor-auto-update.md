# Automatic Vendor File Updates

This project uses Git hooks to automatically keep vendor files synchronized with source code changes.

## What Are Vendor Files?

Vendor files are third-party libraries bundled with the web UI:
- Font Awesome CSS and fonts
- Marked.js (Markdown parser)
- KaTeX (math rendering)
- Marked-KaTeX extension
- Tailwind CSS (compiled from source)

## How It Works

Git hooks automatically run `npm run install:vendor` when HTML, JavaScript, or CSS files change:

### Before Committing (pre-commit)
- Detects changes to web UI files in your commit
- Rebuilds vendor files automatically
- Prompts you to stage vendor files if they changed

### After Switching Branches (post-checkout)
- Detects if web UI files changed between branches
- Automatically rebuilds vendor files

### After Pulling/Merging (post-merge)
- Detects if web UI files changed in the merge
- Automatically rebuilds vendor files

## Setup

Hooks are automatically installed when you run:
```bash
npm install
```

The hooks are version-controlled in the `.githooks/` directory and automatically copied to `.git/hooks/` during installation. This ensures all team members have the same hook configuration.

## Manual Trigger

You can manually rebuild vendor files anytime:
```bash
npm run install:vendor
```

## Temporarily Bypass

To commit without running the pre-commit hook:
```bash
git commit --no-verify
```

## Files Monitored

Hooks monitor changes to:
- `src/abstracts_explorer/web_ui/templates/*.html`
- `src/abstracts_explorer/web_ui/static/*.js`
- `src/abstracts_explorer/web_ui/static/*.css`
- `tailwind.config.js`

## Benefits

✅ Vendor files always match source code  
✅ No manual rebuild steps to remember  
✅ Prevents committing out-of-sync files  
✅ Consistent across team members  

## More Information

See:
- `.githooks/README.md` - Detailed hook documentation and source files
