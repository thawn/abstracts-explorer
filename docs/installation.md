# Installation

## Requirements

- Python 3.8 or higher
- pip package manager
- Node.js 14 or higher (for web UI)
- npm package manager

## Install from Source

Clone the repository and install in development mode:

```bash
git clone <repository-url>
cd neurips-abstracts

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -e .

# Install Node.js dependencies
npm install

# Install vendor files for web UI
npm run install:vendor
```

## Python Dependencies

The package will automatically install the following dependencies:

- **requests**: For API calls to OpenReview
- **chromadb**: For vector embeddings storage
- **pytest**: For running tests
- **pytest-cov**: For test coverage reports
- **pytest-mock**: For test mocking

## Node.js Dependencies

For the web interface, the following dependencies are installed via npm:

- **@fortawesome/fontawesome-free**: Icon library for the UI
- **marked**: Markdown parser for rendering chat responses
- **tailwindcss**: CSS framework (reference)
- **jest**: JavaScript testing framework
- **@testing-library/dom**: Testing utilities for DOM manipulation

### Vendor Files

The web UI uses local copies of external libraries instead of CDN links for:

- **Tailwind CSS** (standalone build): Modern utility-first CSS framework
- **Font Awesome**: Icon fonts and CSS
- **Marked.js**: Markdown parsing and rendering

These files are installed automatically when you run:

```bash
npm run install:vendor
```

This command copies the necessary files from `node_modules` to `src/neurips_abstracts/web_ui/static/vendor/` and `src/neurips_abstracts/web_ui/static/webfonts/`.

**Note**: The vendor files are excluded from git (via `.gitignore`) and must be generated locally or during deployment.

## Optional Dependencies

For documentation building:

```bash
pip install sphinx sphinx-rtd-theme myst-parser sphinx-autodoc-typehints
```

## Verify Installation

Check that the package is installed correctly:

```bash
neurips-abstracts --help
```

You should see the available commands listed.

Test the web UI:

```bash
neurips-abstracts web-ui
```

The web interface should start at <http://127.0.0.1:5000>.

## Troubleshooting

### Missing Vendor Files

If the web UI loads but styling/icons are broken, regenerate vendor files:

```bash
npm run install:vendor
```

### Node.js Not Found

If you don't have Node.js installed:

- **macOS**: `brew install node`
- **Ubuntu/Debian**: `sudo apt install nodejs npm`
- **Windows**: Download from <https://nodejs.org/>

### Python Version Issues

Ensure you're using Python 3.8 or higher:

```bash
python --version
```
