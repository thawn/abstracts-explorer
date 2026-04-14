# Refactoring Plan: De-duplicate and Clean Up the Codebase

This issue contains a detailed plan for de-duplicating and cleaning up the abstracts-explorer codebase. The goal is modular code with atomic single-responsibility functions, simple and clear input arguments, input types, and output types.

**Reference**: See the [Architecture & Flow-Chart documentation](architecture.md) for comprehensive diagrams showing all module relationships and duplicate code paths.

## Analysis Summary

The codebase comprises **~21,800 lines** of Python across 20 source files. Analysis identified **19 distinct patterns of code duplication** spanning all major modules. The most impactful areas for refactoring are:

1. **cli.py** (3,446 lines) — 25 command functions with 8 repeated patterns
2. **database.py** (2,943 lines) — 30+ methods with identical boilerplate
3. **clustering.py** (2,094 lines) — duplicated TF-IDF and LLM label generation
4. **mcp_server.py + mcp_tools.py + rag.py** (3,026 lines) — duplicated tool patterns across 3 modules
5. **web_ui/app.py** (1,120 lines) — 22 routes with identical error handling

---

## Phase 1: CLI De-duplication (cli.py — HIGH IMPACT)

The CLI module has 8 major repeated patterns across 25 command functions.

### Step 1.1: Extract EmbeddingsManager Factory Function

**Current state**: 6 commands repeat identical EmbeddingsManager initialization:
```python
config = get_config()
em = EmbeddingsManager(
    lm_studio_url=config.llm_backend_url,
    auth_token=config.llm_backend_auth_token,
    model_name=config.embedding_model,
    collection_name=config.collection_name,
    requests_per_minute=config.requests_per_minute,
)
em.connect()
em.create_collection()
```

**Action**:
- [ ] Create `_init_embeddings_manager(config: Config) -> EmbeddingsManager` helper function
- [ ] Replace 6 occurrences in: `create_embeddings_command`, `search_command`, `chat_command`, `eval_generate_command`, `eval_run_command`, `pre_generate_clustering_command`
- [ ] Add tests for the new helper function

### Step 1.2: Extract Embeddings Validation Helper

**Current state**: 4 commands repeat embeddings DB existence check with identical error messages.

**Action**:
- [ ] Create `_validate_embeddings_db(config: Config) -> bool` helper that checks path and prints error
- [ ] Replace 4 occurrences in: `search_command`, `chat_command`, `cluster_embeddings_command`, `pre_generate_clustering_command`

### Step 1.3: Extract LM Studio Connection Test Helper

**Current state**: 4 commands repeat `em.test_lm_studio_connection()` with identical error messages.

**Action**:
- [ ] Create `_test_api_connection(em: EmbeddingsManager) -> bool` helper
- [ ] Replace 4 occurrences in: `create_embeddings_command`, `search_command`, `chat_command`, `eval_run_command`

### Step 1.4: Extract Command Header Printer

**Current state**: 15+ commands print 70-char-width headers with identical formatting.

**Action**:
- [ ] Create `_print_command_header(title: str, **params)` utility function
- [ ] Replace all header printing blocks across all commands

### Step 1.5: Extract Confirmation Prompt Helper

**Current state**: 3 commands repeat `--yes` flag check + `input()` confirmation pattern.

**Action**:
- [ ] Create `_confirm_action(message: str, args: Namespace) -> bool` helper
- [ ] Replace 3 occurrences in: `delete_data_command`, `eval_results_command`, `registry_delete_command`

### Step 1.6: Standardize Error Handling

**Current state**: 15+ commands repeat `try/except` with `traceback.print_exc()` and similar error messages.

**Action**:
- [ ] Consider a command decorator that wraps error handling
- [ ] Alternatively, extract `_handle_command_error(e: Exception, command_name: str) -> int` helper
- [ ] Standardize exit codes and error message format

---

## Phase 2: Database Layer De-duplication (database.py — HIGH IMPACT)

### Step 2.1: Eliminate Session Validation Boilerplate

**Current state**: 30+ methods start with `if not self._session: raise DatabaseError(...)`.

**Action**:
- [ ] Create a `_require_session(self)` method or use a decorator `@require_session`
- [ ] Replace all 30+ occurrences of the inline check
- [ ] Ensure consistent error message format

### Step 2.2: Remove Duplicate Faceting Method

**Current state**: `get_years_for_conference(conference)` duplicates `get_years(conference=conf)`.

**Action**:
- [ ] Deprecate or remove `get_years_for_conference()`
- [ ] Update all callers to use `get_years(conference=...)` instead
- [ ] Search for callers in: `registry.py`, `web_ui/app.py`, `cli.py`

### Step 2.3: Extract Generic Faceting Method

**Current state**: `get_sessions()`, `get_conferences()`, `get_years()` follow identical patterns.

**Action**:
- [ ] Create `_get_distinct_values(column, filters: Dict) -> List` generic method
- [ ] Refactor 3 faceting methods to use the generic method
- [ ] Keep public API signatures unchanged (backward compatible)

### Step 2.4: Consolidate Eval CRUD Operations

**Current state**: EvalQAPair and EvalResult CRUD methods follow parallel structures.

**Action**:
- [ ] Evaluate if a generic CRUD helper would reduce duplication
- [ ] If justified, create `_add_record()`, `_get_records()`, `_delete_records()` helpers
- [ ] Lower priority — each method has specific query logic

---

## Phase 3: Clustering De-duplication (clustering.py — MEDIUM IMPACT)

### Step 3.1: Consolidate TF-IDF Keyword Extraction

**Current state**: `extract_cluster_keywords()` and `_extract_keywords_for_samples()` contain nearly identical TF-IDF logic (~95% code similarity).

**Action**:
- [ ] Extract common TF-IDF logic to `_compute_tfidf_keywords(documents: List[str], n_keywords: int, min_df: int) -> List[str]`
- [ ] Refactor both methods to call the shared implementation
- [ ] Update tests to cover the new shared function

### Step 3.2: Consolidate LLM Label Generation

**Current state**: 3 methods use identical OpenAI API call patterns:
- `_generate_llm_label(cluster_id, keywords)`
- `_generate_parent_label_llm(child_labels, sample_indices)`
- `_generate_llm_label_from_keywords(keywords)`

**Action**:
- [ ] Extract `_call_llm_for_label(prompt: str, fallback: str) -> str` helper
- [ ] Refactor all 3 methods to build their specific prompt and call the shared helper
- [ ] Centralize error handling and fallback logic

### Step 3.3: Extract ChromaDB Where-clause Builder

**Current state**: Where-clause construction for conference/year filtering is duplicated in:
- `embeddings.py`: `search_papers_semantic()`, `find_papers_within_distance()`
- `clustering.py`: `load_embeddings()`, `compute_clusters_with_cache()`
- `mcp_server.py`: `merge_where_clause_with_conference()`, `merge_where_clause_with_years()`

**Action**:
- [ ] Create `build_chromadb_where_clause(conferences: List[str], years: List[int], base_where: Dict = None) -> Dict` in a shared utility (e.g., `embeddings.py` or new `chroma_utils.py`)
- [ ] Replace all 5+ occurrences across 3 modules
- [ ] Move `merge_where_clause_with_conference/years` from `mcp_server.py` to shared utility
- [ ] Add comprehensive tests for edge cases (empty lists, None values, nested $and)

---

## Phase 4: MCP/RAG Tool De-duplication (MEDIUM IMPACT)

### Step 4.1: Consolidate RAG Tool Wrapper Functions

**Current state**: 6 `_tool_*()` functions in `rag.py` follow identical patterns:
```python
async def _tool_wrapper(ctx, **args):
    result = mcp_function(**args)
    logger.debug(...)
    ctx.deps.tool_results.append({"tool": name, "raw_result": result})
    return format_tool_result_for_llm(name, result)
```

**Action**:
- [ ] Create a generic tool wrapper factory: `_make_tool_wrapper(tool_name: str, mcp_func: Callable) -> Callable`
- [ ] Generate all 6 wrappers programmatically
- [ ] Preserve function signatures for Pydantic AI introspection (use `functools.wraps` or explicit parameter definitions)

### Step 4.2: Consolidate MCP Argument Normalization

**Current state**: 4 normalization functions in `mcp_tools.py` fix similar LLM quirks (singular→plural, list→scalar, string→type).

**Action**:
- [ ] Create data-driven normalization: define transformation rules as declarative config
- [ ] Create `_normalize_args(arguments: Dict, rules: List[NormRule]) -> Dict` generic function
- [ ] Define per-tool rules instead of per-tool functions
- [ ] Reduce ~120 lines of near-duplicate code to ~40 lines

### Step 4.3: Consolidate MCP Resource Initialization

**Current state**: 6 tool functions in `mcp_server.py` each independently create `EmbeddingsManager` and `DatabaseManager`.

**Action**:
- [ ] Create `_get_mcp_resources(collection_name: str) -> Tuple[EmbeddingsManager, DatabaseManager]` helper with proper cleanup
- [ ] Use context manager pattern for automatic resource cleanup
- [ ] Replace 6 occurrences of init/cleanup boilerplate

---

## Phase 5: Web UI Route De-duplication (LOW-MEDIUM IMPACT)

### Step 5.1: Extract Common Error Handler Decorator

**Current state**: All 22 routes wrap logic in `try/except` with `logger.error()` and `jsonify()`.

**Action**:
- [ ] Create `@api_error_handler` decorator that catches exceptions, logs them, and returns jsonified error responses
- [ ] Apply decorator to all API routes
- [ ] Preserve route-specific error handling where needed (e.g., 404 vs 500)

### Step 5.2: Extract Parameter Validation Helpers

**Current state**: Multiple routes manually validate and convert request parameters.

**Action**:
- [ ] Create `get_int_param(request, name, default=None)` and similar helpers
- [ ] Create `get_conference_year_params(request) -> Tuple[str, int]` for the common pattern
- [ ] Replace manual parameter parsing in routes

---

## Phase 6: Export & Utility De-duplication (LOW IMPACT)

### Step 6.1: Consolidate Paper Markdown Formatting

**Current state**: `generate_all_papers_markdown()` and `generate_search_term_markdown()` in `export_utils.py` share ~80% identical code for session grouping and paper block formatting.

**Action**:
- [ ] Extract `_format_paper_block(paper: Dict) -> str` helper
- [ ] Extract `_group_papers_by_session(papers: List[Dict]) -> Dict[str, List[Dict]]` helper
- [ ] Refactor both functions to use shared helpers
- [ ] Keep public API signatures unchanged

### Step 6.2: Consolidate Serialization Functions (plugin.py)

**Current state**: `serialize_authors_to_string()` and `serialize_keywords_to_string()` follow identical patterns.

**Action**:
- [ ] Consider a generic `_serialize_list_to_string(items: List[str], separator: str) -> str`
- [ ] Lower priority — the functions are only 2-3 lines each

---

## Phase 7: Registry Cleanup (LOW IMPACT)

### Step 7.1: Consolidate Progress Callback Pattern

**Current state**: 4 methods in `registry.py` define identical `_progress()` inner functions.

**Action**:
- [ ] Create `_make_progress_callback(progress_callback, phase_name) -> Callable` factory
- [ ] Replace 4 identical inner function definitions

---

## Implementation Guidelines

### Testing Requirements
- Every refactored function must maintain its existing tests passing
- New helper functions must have dedicated unit tests
- Run `uv run pytest tests/ -x --ignore=tests/test_web_e2e.py --ignore=tests/test_web_integration.py --ignore=tests/test_integration.py --ignore=tests/test_staging_e2e.py --ignore=tests/test_registry_integration.py -q` after each step

### Code Standards
- All new functions must have type hints and NumPy-style docstrings
- Line length: 118 characters (Black formatter)
- No change to public API signatures (backward compatibility)
- Follow existing exception hierarchy patterns

### Priority Order
1. **Phase 3.3** (ChromaDB where-clause) — cross-cutting, affects 3 modules
2. **Phase 1.1-1.3** (CLI helpers) — largest file, most repetition
3. **Phase 2.1** (Session validation) — 30+ occurrences
4. **Phase 3.1-3.2** (Clustering TF-IDF & LLM) — medium duplication
5. **Phase 4.1-4.3** (MCP/RAG tools) — medium duplication
6. **Phase 5.1** (Web error handler) — clean improvement
7. **Remaining phases** — lower impact

### Estimated Impact
- **Lines removed (estimated)**: ~800-1200 lines of duplicated code
- **New helper functions**: ~15-20 atomic, well-tested utilities
- **Modules affected**: All 20 source files
- **Risk level**: Low-Medium (refactoring internal helpers, public API unchanged)

---

_This plan was created by analyzing the full codebase architecture. See [architecture.md](architecture.md) for comprehensive flow-charts and module diagrams._

Closes #306
