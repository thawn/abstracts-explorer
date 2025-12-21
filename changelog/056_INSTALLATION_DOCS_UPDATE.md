# Documentation: Node.js Dependencies and Vendor Files

**Date**: November 27, 2025

## Summary

Updated installation documentation in both `README.md` and `docs/installation.md` to include instructions for installing Node.js dependencies and generating vendor files required by the web UI.

## Changes

### README.md

Updated the **Installation** section to include:

1. **Node.js requirement** added to requirements list
2. **npm install** step for Node.js dependencies
3. **npm run install:vendor** step to generate vendor files
4. Added **Start Web Interface** subsection under CLI commands

#### Installation Section Enhancement

```bash
# Install Node.js dependencies for web UI
npm install

# Install vendor files (Tailwind CSS, Font Awesome, Marked.js)
npm run install:vendor
```

#### Requirements Updated

- Python 3.8+
- **Node.js 14+** (for web UI) ← NEW
- requests >= 2.31.0

#### New Web UI CLI Documentation

Added comprehensive documentation for the `web-ui` command:
- Basic usage examples
- Custom host/port configuration
- Database and embeddings path specification
- Debug mode
- Feature list (Search, Chat, Filters, Details)

### docs/installation.md

Major enhancement to the installation documentation:

#### Added Node.js Requirements

- Node.js 14 or higher
- npm package manager

#### Expanded Install Steps

Complete installation workflow now includes:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -e .

# Install Node.js dependencies
npm install

# Install vendor files for web UI
npm run install:vendor
```

#### New Section: Node.js Dependencies

Documented the npm packages installed:
- `@fortawesome/fontawesome-free` - Icon library
- `marked` - Markdown parser for chat
- `tailwindcss` - CSS framework reference
- `jest` - JavaScript testing framework
- `@testing-library/dom` - Testing utilities

#### New Section: Vendor Files

Explained the vendor file system:
- What vendor files are (Tailwind CSS, Font Awesome, Marked.js)
- Why they're used (local copies instead of CDN)
- How to install them (`npm run install:vendor`)
- Where they're installed (`static/vendor/`, `static/webfonts/`)
- Git exclusion note (files are in `.gitignore`)

#### New Section: Verify Installation - Web UI

Added web UI verification:

```bash
neurips-abstracts web-ui
```

Expected result: Server starts at <http://127.0.0.1:5000>

#### New Section: Troubleshooting

Added troubleshooting guide:

**Missing Vendor Files**
- Symptom: Broken styling/icons
- Solution: `npm run install:vendor`

**Node.js Not Found**
- Installation instructions for macOS, Ubuntu/Debian, Windows

**Python Version Issues**
- How to check Python version

## Rationale

### User Confusion Prevention

Without clear documentation, new users might:
- Clone the repo and miss npm installation
- Wonder why the web UI has broken styling
- Not understand the vendor file system
- Skip Node.js thinking it's optional

### Complete Setup Workflow

Provides a single source of truth for:
- All required dependencies (Python and Node.js)
- Correct installation order
- Verification steps
- Common issues and solutions

### Web UI Visibility

The web UI is a major feature that deserves:
- Clear installation instructions
- Prominent placement in CLI documentation
- Feature list to explain capabilities
- Usage examples

## Benefits

**Clear Installation Path**
Users know exactly what to install and in what order.

**Reduced Support Questions**
Common issues are documented with solutions.

**Feature Discovery**
Users learn about the web UI and its capabilities.

**Platform-Specific Help**
Troubleshooting section covers macOS, Linux, and Windows.

**Professional Documentation**
Complete installation guide increases project credibility.

## Files Modified

- `README.md` - Updated Installation and CLI sections
- `docs/installation.md` - Comprehensive installation guide

## Documentation Structure

### README.md (Quick Start)

- Brief installation steps
- Requirements list
- Web UI CLI command examples
- Feature list

### docs/installation.md (Detailed)

- Complete installation workflow
- Detailed dependency explanations
- Vendor file system explanation
- Verification procedures
- Troubleshooting guide

## Testing

Verified that:
- ✅ All commands are correct
- ✅ File paths are accurate
- ✅ URLs are valid
- ✅ Instructions work on the actual system

## Related Changes

- See `changelog/53_LOCAL_VENDOR_FILES.md` - Local vendor implementation
- See `changelog/54_CLEAN_PACKAGE_LOCK.md` - Package lock recreation
- See `changelog/55_GITIGNORE_VENDOR_FILES.md` - Git ignore setup

## Future Improvements

- Could add video tutorial for installation
- May want to create a setup script that runs all commands
- Could add Docker installation option
- Consider adding CI/CD deployment documentation
