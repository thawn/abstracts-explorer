# NeurIPS Abstracts

A Python package for downloading NeurIPS conference data and loading it into a SQLite database.

## Features

- 📥 Download NeurIPS conference data from configurable URLs
- 💾 Store data in a SQLite database with efficient indexing
- 🔍 Search and query papers by keywords, track, and other attributes
- 🤖 **NEW**: Generate text embeddings and store in vector database for semantic search
- 🔎 **NEW**: Find similar papers using AI-powered semantic similarity
- 💬 **NEW**: Interactive RAG chat to ask questions about papers
- ⚙️ **NEW**: Environment-based configuration with `.env` file support
- 🧪 Comprehensive test suite with pytest (123 tests, 78% coverage)
- 📚 Full documentation with NumPy-style docstrings

## Installation

### From source

```bash
# Clone the repository
git clone https://github.com/yourusername/neurips-abstracts.git
cd neurips-abstracts

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install Node.js dependencies for web UI
npm install

# Install vendor files (Tailwind CSS, Font Awesome, Marked.js)
npm run install:vendor
```

### Requirements

- Python 3.8+
- Node.js 14+ (for web UI)
- requests >= 2.31.0

## Configuration

The package supports environment-based configuration using `.env` files. This allows you to customize default settings without modifying command-line arguments.

### Quick Setup

```bash
# Copy the example configuration file
cp .env.example .env

# Edit with your preferred settings
nano .env
```

### Available Settings

- `DATA_DIR` - Base directory for data files (default: data)
- `CHAT_MODEL` - Language model for RAG chat (default: diffbot-small-xl-2508)
- `EMBEDDING_MODEL` - Text embedding model (default: text-embedding-qwen3-embedding-4b)
- `LLM_BACKEND_URL` - LM Studio API URL (default: http://localhost:1234)
- `LLM_BACKEND_AUTH_TOKEN` - Authentication token (optional)
- `EMBEDDING_DB_PATH` - ChromaDB directory (default: chroma_db, resolved relative to DATA_DIR)
- `PAPER_DB_PATH` - SQLite database (default: neurips_2025.db, resolved relative to DATA_DIR)
- `COLLECTION_NAME` - ChromaDB collection name (default: neurips_papers)
- `MAX_CONTEXT_PAPERS` - Papers for RAG context (default: 5)
- `ENABLE_QUERY_REWRITING` - Enable AI-powered query rewriting (default: true)
- `QUERY_SIMILARITY_THRESHOLD` - Similarity threshold for caching (default: 0.7)

See [CONFIGURATION.md](CONFIGURATION.md) for complete documentation.

## Quick Start

### Download NeurIPS Data

```python
from neurips_abstracts import download_json, download_neurips_data

# Download from a specific URL
data = download_json("https://neurips.cc/static/virtual/data/neurips-2025-orals-posters.json")

# Or use the convenience function
data = download_neurips_data(year=2025)

# Save to a file
data = download_json(
    "https://neurips.cc/static/virtual/data/neurips-2025-orals-posters.json",
    output_path="data/neurips_2025.json"
)
```

### Load Data into Database

```python
from neurips_abstracts import DatabaseManager

# Create and connect to database
with DatabaseManager("neurips.db") as db:
    # Create tables
    db.create_tables()
    
    # Load JSON data
    count = db.load_json_data(data)
    print(f"Loaded {count} papers")
    
    # Get total paper count
    total = db.get_paper_count()
    print(f"Total papers in database: {total}")
```

### Search Papers

```python
from neurips_abstracts import DatabaseManager

with DatabaseManager("neurips.db") as db:
    # Search by keyword
    papers = db.search_papers(keyword="neural network")
    
    # Search by event type
    oral_papers = db.search_papers(eventtype="Oral")
    
    # Search by decision
    poster_papers = db.search_papers(decision="Accept (poster)")
    
    # Search by topic
    ml_papers = db.search_papers(topic="Machine Learning")
    
    # Combined search with limit
    papers = db.search_papers(
        keyword="reinforcement learning",
        eventtype="Poster",
        limit=10
    )
    
    # Display results
    for paper in papers:
        print(f"{paper['name']} - {paper['authors']}")
```

### Query Authors

```python
from neurips_abstracts import DatabaseManager

with DatabaseManager("neurips.db") as db:
    # Search authors by name
    authors = db.search_authors(name="Huang")
    for author in authors:
        print(f"{author['fullname']} - {author['institution']}")
    
    # Search by institution
    stanford_authors = db.search_authors(institution="Stanford")
    
    # Get all papers by a specific author
    papers = db.get_author_papers(author_id=457880)
    print(f"Found {len(papers)} papers by this author")
    
    # Get all authors for a specific paper
    authors = db.get_paper_authors(paper_id=123456)
    for author in authors:
        print(f"{author['author_order']}. {author['fullname']}")
    
    # Get author count
    count = db.get_author_count()
    print(f"Total unique authors: {count}")
```

### Custom Queries

```python
from neurips_abstracts import DatabaseManager

with DatabaseManager("neurips.db") as db:
    # Execute custom SQL queries
    results = db.query(
        "SELECT name, authors, decision FROM papers WHERE eventtype = ? ORDER BY name",
        ("Oral",)
    )
    
    for row in results:
        print(f"{row['name']}: {row['authors']} ({row['decision']})")
```

## Complete Example

Here's a complete example that downloads NeurIPS 2025 data and loads it into a database:

```python
from neurips_abstracts import download_neurips_data, DatabaseManager

# Download data
print("Downloading NeurIPS 2025 data...")
data = download_neurips_data(
    year=2025,
    output_path="data/neurips_2025.json"
)

# Load into database
print("Loading data into database...")
with DatabaseManager("neurips.db") as db:
    db.create_tables()
    count = db.load_json_data(data)
    print(f"Loaded {count} papers")
    
    # Search for papers about deep learning
    papers = db.search_papers(keyword="deep learning", limit=5)
    
    print(f"\nFound {len(papers)} papers about deep learning:")
    for paper in papers:
        print(f"- {paper['name']}")
        print(f"  Authors: {paper['authors']}")
        print(f"  Decision: {paper['decision']}")
        print(f"  Session: {paper['session']}")
        print()
```

## Command-Line Interface

The package includes a powerful CLI for common tasks:

### Download Data

```bash
# Download NeurIPS 2025 data and create database
neurips-abstracts download --year 2025 --output neurips_2025.db

# Force re-download
neurips-abstracts download --year 2025 --output neurips_2025.db --force
```

### Generate Embeddings

```bash
# Generate embeddings for all papers
neurips-abstracts create-embeddings --db-path neurips_2025.db

# Use custom output directory
neurips-abstracts create-embeddings \
  --db-path neurips_2025.db \
  --output embeddings/ \
  --collection neurips_2025

# Generate embeddings only for accepted papers
neurips-abstracts create-embeddings \
  --db-path neurips_2025.db \
  --where "decision LIKE '%Accept%'"

# Use custom settings
neurips-abstracts create-embeddings \
  --db-path neurips_2025.db \
  --batch-size 50 \
  --lm-studio-url http://localhost:5000 \
  --model custom-embedding-model
```

### Start Web Interface

```bash
# Start the web UI with default settings
neurips-abstracts web-ui

# Use custom host and port
neurips-abstracts web-ui --host 0.0.0.0 --port 8080

# Specify database and embeddings location
neurips-abstracts web-ui \
  --db-path neurips_2025.db \
  --embeddings-path chroma_db

# Enable debug mode
neurips-abstracts web-ui --debug
```

The web interface provides:
- 🔍 **Search**: Keyword and AI-powered semantic search
- 💬 **Chat**: Interactive RAG chat to ask questions about papers
  - ✨ **NEW**: Displays rewritten query showing how your question was optimized
  - 📊 Cache status indicator (retrieved vs. cached papers)
- 📊 **Filters**: Filter by track, decision, event type, session, and topics
- 📄 **Details**: View full paper information including authors and abstracts

See `CLI_REFERENCE.md` for complete CLI documentation and examples.

## Semantic Search with Embeddings

**NEW**: Generate text embeddings and perform semantic similarity search!

### Prerequisites

1. Install [LM Studio](https://lmstudio.ai/) and load the `text-embedding-qwen3-embedding-4b` model
2. Start the LM Studio server (default: http://localhost:1234)
3. Install ChromaDB: `pip install chromadb`

### Generate Embeddings

```python
from neurips_abstracts import EmbeddingsManager

# Initialize embeddings manager
with EmbeddingsManager() as em:
    em.create_collection()
    
    # Embed papers from database
    count = em.embed_from_database(
        "neurips.db",
        where_clause="decision = 'Accept'"  # Optional filter
    )
    print(f"Embedded {count} papers")
```

### Search Similar Papers

```python
from neurips_abstracts import EmbeddingsManager

with EmbeddingsManager() as em:
    em.create_collection()
    
    # Find papers similar to a query
    results = em.search_similar(
        "deep learning transformers for natural language processing",
        n_results=5
    )
    
    # Display results
    for i, paper_id in enumerate(results['ids'][0], 1):
        metadata = results['metadatas'][0][i-1]
        similarity = 1 - results['distances'][0][i-1]
        print(f"{i}. {metadata['title']}")
        print(f"   Similarity: {similarity:.4f}")
        print(f"   Authors: {metadata['authors']}")
        print()
```

See `EMBEDDINGS_MODULE.md` for complete documentation and `examples/embeddings_demo.py` for a full demonstration.

## Database Schema

The package creates three tables with proper relational design:

### Papers Table

| Column          | Type      | Description                                         |
| --------------- | --------- | --------------------------------------------------- |
| id              | INTEGER   | Primary key - paper ID                              |
| uid             | TEXT      | Unique hash identifier                              |
| name            | TEXT      | Paper title                                         |
| abstract        | TEXT      | Paper abstract                                      |
| authors         | TEXT      | Comma-separated author IDs (links to authors table) |
| keywords        | TEXT      | Comma-separated list of keywords                    |
| topic           | TEXT      | Research topic/category                             |
| decision        | TEXT      | Acceptance decision (e.g., "Accept (poster)")       |
| session         | TEXT      | Session name                                        |
| eventtype       | TEXT      | Event type (Poster, Oral, etc.)                     |
| event_type      | TEXT      | Event type template                                 |
| room_name       | TEXT      | Physical location                                   |
| virtualsite_url | TEXT      | Virtual conference URL                              |
| paper_url       | TEXT      | OpenReview paper URL                                |
| starttime       | TEXT      | Event start time (ISO 8601)                         |
| endtime         | TEXT      | Event end time (ISO 8601)                           |
| poster_position | TEXT      | Poster location                                     |
| raw_data        | TEXT      | Complete JSON data as text                          |
| created_at      | TIMESTAMP | Record creation timestamp                           |
| ...             | ...       | Plus 20+ more fields                                |

### Authors Table

| Column      | Type      | Description                          |
| ----------- | --------- | ------------------------------------ |
| id          | INTEGER   | Primary key - author ID from NeurIPS |
| fullname    | TEXT      | Full name of the author              |
| url         | TEXT      | NeurIPS API URL for author details   |
| institution | TEXT      | Author's institution                 |
| created_at  | TIMESTAMP | Record creation timestamp            |

### Paper-Authors Junction Table

| Column       | Type    | Description                             |
| ------------ | ------- | --------------------------------------- |
| paper_id     | INTEGER | Foreign key to papers.id                |
| author_id    | INTEGER | Foreign key to authors.id               |
| author_order | INTEGER | Order of author in paper (1, 2, 3, ...) |

The junction table enables many-to-many relationships between papers and authors, preserving author order.

Indexes are created on commonly queried fields for efficient searches. See `SCHEMA_MIGRATION.md` for complete details.

## Configuration

### Custom Download URL

You can download from any URL that returns JSON data:

```python
from neurips_abstracts import download_json

# Custom URL
data = download_json("https://your-custom-url.com/data.json")

# With custom timeout and SSL verification
data = download_json(
    "https://your-custom-url.com/data.json",
    timeout=60,
    verify_ssl=False
)
```

### Database Location

Specify any path for your database:

```python
from neurips_abstracts import DatabaseManager

# Use a specific path
db = DatabaseManager("/path/to/your/database.db")

# Or relative path
db = DatabaseManager("data/neurips.db")
```

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/neurips-abstracts.git
cd neurips-abstracts

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests (excluding slow tests by default)
pytest

# Run with coverage report
pytest --cov=neurips_abstracts --cov-report=html

# Run specific test file
pytest tests/test_downloader.py

# Run specific test
pytest tests/test_database.py::TestDatabaseManager::test_connect

# Run only slow tests (requires LM Studio running)
pytest -m slow

# Run all tests including slow ones
pytest -m ""

# Run end-to-end tests (requires Chrome or Firefox browser)
pytest -m e2e

# Run E2E tests with verbose output
pytest tests/test_web_e2e.py -v -m e2e

# Run E2E tests with Firefox instead of Chrome
E2E_BROWSER=firefox pytest tests/test_web_e2e.py -v -m e2e
```

**Note:**
- Tests requiring LM Studio are marked as `slow` and skipped by default. To run them, use `pytest -m slow` (requires LM Studio running with a chat model loaded).
- End-to-end tests are marked as `e2e` and require either Chrome or Firefox browser. These tests use Selenium to automate browser interactions and verify the web UI works correctly. By default, Chrome is tried first, then Firefox. You can specify a browser with the `E2E_BROWSER` environment variable.

### Code Structure

```
neurips-abstracts/
├── src/
│   └── neurips_abstracts/
│       ├── __init__.py         # Package initialization
│       ├── downloader.py       # Download functionality
│       └── database.py         # Database management
├── tests/
│   ├── __init__.py
│   ├── test_downloader.py      # Downloader tests
│   ├── test_database.py        # Database tests
│   └── test_integration.py     # Integration tests
├── pyproject.toml              # Package configuration
└── README.md                   # This file
```

## Error Handling

The package provides custom exceptions for better error handling:

```python
from neurips_abstracts import download_json, DatabaseManager
from neurips_abstracts.downloader import DownloadError
from neurips_abstracts.database import DatabaseError

try:
    data = download_json("https://invalid-url.com/data.json")
except DownloadError as e:
    print(f"Download failed: {e}")

try:
    with DatabaseManager("neurips.db") as db:
        db.load_json_data(invalid_data)
except DatabaseError as e:
    print(f"Database error: {e}")
```

## Logging

The package uses Python's built-in logging. Configure it to see detailed logs:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.INFO)

# Now use the package
from neurips_abstracts import download_neurips_data
data = download_neurips_data()
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For issues, questions, or contributions, please visit:
https://github.com/yourusername/neurips-abstracts/issues

## Query Rewriting Feature

The RAG system now includes intelligent query rewriting to improve search results:

- **Automatic Query Optimization**: User questions are automatically rewritten into effective search queries using the LLM
- **Context-Aware Rewriting**: Follow-up questions consider conversation history for better context
- **Smart Caching**: Similar follow-up queries reuse cached papers to reduce unnecessary retrievals
- **Configurable**: Enable/disable via `ENABLE_QUERY_REWRITING` environment variable
- **Tunable Threshold**: Control caching behavior with `QUERY_SIMILARITY_THRESHOLD` (0.0-1.0)

Example:
```python
from neurips_abstracts import RAGChat, EmbeddingsManager, DatabaseManager

with EmbeddingsManager() as em, DatabaseManager("neurips.db") as db:
    chat = RAGChat(em, db)
    
    # First query - rewrites and retrieves papers
    response1 = chat.query("What about transformers?")
    # Rewritten: "transformer architecture attention mechanism neural networks"
    
    # Follow-up - detects similar query, reuses cached papers
    response2 = chat.query("Tell me more about transformers")
    # Reuses same papers without re-retrieval
    
    # Different topic - retrieves new papers
    response3 = chat.query("What about reinforcement learning?")
    # New retrieval for different topic
```

## ToDo

- Further RAG improvements
  - consider [multi-turn conversation refinement](https://www.emergentmind.com/topics/multi-turn-rag-conversations)
  - Implement citation extraction and validation
