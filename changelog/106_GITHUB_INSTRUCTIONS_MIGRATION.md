# Migration to GitHub Copilot Repository Instructions Format

## Summary

Migrated `.instructions.md` files to follow GitHub's official documentation for repository custom instructions.

## Changes Made

### File Locations

**Before:**

- `/.instructions.md` (root level)
- `/changelog/.instructions.md`

**After:**

- `/.github/instructions/project.instructions.md`
- `/.github/instructions/changelog.instructions.md`

### Format Updates

All instruction files now follow GitHub's path-specific custom instructions format:

1. **Created `.github/instructions/` directory** - Official location for path-specific instructions
2. **Added frontmatter blocks** with `applyTo` glob patterns:
   - `project.instructions.md`: `applyTo: "**"` (applies to all files)
   - `changelog.instructions.md`: `applyTo: "changelog/**/*.md"` (applies to changelog files only)
3. **Proper naming convention** - Files end with `.instructions.md`

## Format Specification

According to [GitHub's documentation](https://docs.github.com/en/copilot/customizing-copilot/adding-custom-instructions-for-github-copilot):

### Path-Specific Instructions Format

```markdown
---
applyTo: "glob/pattern/**/*.ext"
---

# Your instructions here
```

### Key Requirements

- Files must be in `.github/instructions/` directory
- Filename must end with `.instructions.md`
- Must include frontmatter block with `applyTo` keyword
- Use glob syntax to specify which files the instructions apply to

### Glob Pattern Examples

- `**` - All files in all directories
- `**/*.py` - All Python files recursively
- `src/**/*.ts` - All TypeScript files in src directory
- `changelog/**/*.md` - All Markdown files in changelog directory

## Benefits

1. **Official Format**: Follows GitHub Copilot's documented standard
2. **Better Organization**: Centralized in `.github/instructions/`
3. **Scoped Instructions**: Different instructions can apply to different file types/locations
4. **Tool Support**: Works properly with GitHub Copilot features including:
   - Copilot Chat
   - Copilot coding agent
   - Copilot code review

## Testing

The new instruction files are ready to use. Copilot will automatically:

- Detect instructions in `.github/instructions/`
- Apply them based on `applyTo` glob patterns
- Include them in relevant responses

## References

- [GitHub Docs: Adding repository custom instructions](https://docs.github.com/en/copilot/customizing-copilot/adding-custom-instructions-for-github-copilot)
- [Custom Instructions Examples](https://docs.github.com/en/copilot/tutorials/customization-library/custom-instructions)

---

**Date**: December 21, 2025
