# Codebase Architecture & Flow-Charts

This document provides a comprehensive architectural overview of the Abstracts Explorer
codebase, including module relationships, data flow, and function-level flow-charts.

## Module Overview

The codebase comprises **~21,200 lines** of Python (plus the dev-only
`tools/profiling.py`, which is not shipped in the wheel):

| Module | Lines | Responsibility |
|--------|------:|----------------|
| `cli/` package | 4,640 | Command-line interface (`main.py` dispatch + `parsers.py` wiring + per-command modules) |
| `db/` package | 4,061 | SQL database operations as `DatabaseManager` mixins (+ 27-line `database.py` facade) |
| `plugins/` | 4,079 | Conference downloader plugins (11 plugins) |
| `clustering/` package | 1,976 | Agglomerative clustering, t-SNE, labeling, hierarchy, cache |
| `plugin/` package | 1,753 | Plugin framework, models, registry, HTTP, persistence |
| `registry.py` | 1,684 | OCI registry upload/download |
| `mcp/` package | 1,481 | MCP server, resources, cache, tool implementations (8 tools) |
| `embeddings/` package | 1,426 | ChromaDB store, generation, queries, ingestion, artifacts |
| `rag/` package | 1,064 | RAG agent, tool wrappers, conversation, extraction |
| `web_ui/app.py` | 1,037 | Flask web API (19 routes) |
| `web_ui/dependencies.py` | 178 | Request-scoped DB/embeddings/RAG helpers |
| `evaluation/` package | 882 | RAG evaluation (prompts, generation, running) |
| `interesting_papers/` package | 823 | Interesting-paper summaries & recommendations |
| `export/` package | 709 | Paper export (ZIP/Markdown) |
| `mcp_tools.py` | 705 | Tool registry, argument normalization, LLM formatting |
| `config.py` | 501 | Configuration management |
| `db_models.py` | 429 | SQLAlchemy ORM models |
| `utils/` package | 256 | Shared time/json/url/conference/presentation helpers |
| `rate_limiter.py` | 222 | Token-bucket rate limiter & cooldown |
| `paper_utils.py` | 167 | Paper formatting utilities |
| `metadata_cache.py` | 112 | TTL cache for PostgreSQL metadata |
| `llm_utils.py` | 76 | Shared one-shot chat-completion helper |
| `tools/profiling.py` | 745 | DB profiling (development tool, not shipped) |

## High-Level Module Dependency Graph

```{mermaid}
graph TD
    CLI[cli/ package<br/>15 commands<br/>+ subcommand groups]
    CFG[config.py<br/>501 lines]
    DB[db/ package<br/>DatabaseManager mixins]
    DBM[db_models.py<br/>429 lines]
    EMB[embeddings/<br/>ChromaDB + generation]
    CLU[clustering/<br/>Agglomerative + t-SNE]
    RAG[rag/<br/>agent + tools]
    MCP[mcp/<br/>server + tools]
    MCT[mcp_tools.py<br/>registry + formatting]
    WEB[web_ui/<br/>Flask app + deps]
    PLG[plugin/<br/>framework + registry]
    PLI[plugins/*<br/>11 downloaders]
    EXP[export/<br/>markdown + zip]
    PAP[paper_utils.py<br/>167 lines]
    REG[registry.py<br/>1684 lines]
    EVL[evaluation/<br/>generate + run]

    CLI --> CFG
    CLI --> DB
    CLI --> EMB
    CLI --> CLU
    CLI --> RAG
    CLI --> MCP
    CLI --> PLG
    CLI --> EVL
    CLI --> WEB
    CLI --> REG

    WEB --> DB
    WEB --> EMB
    WEB --> RAG
    WEB --> CFG
    WEB --> PAP
    WEB --> EXP
    WEB --> CLU

    RAG --> CFG
    RAG --> MCP
    RAG --> MCT

    MCP --> EMB
    MCP --> DB
    MCP --> CLU
    MCP --> CFG

    MCT --> MCP

    EMB --> CFG
    EMB --> DB
    EMB --> PLG
    EMB --> PAP

    CLU --> EMB
    CLU --> DB
    CLU --> CFG

    DB --> CFG
    DB --> PLG
    DB --> DBM

    REG --> DB
    REG --> EMB
    REG --> DBM
    REG --> CFG

    EVL --> CFG
    EVL --> DB
    EVL --> EMB
    EVL --> MCT
    EVL --> RAG

    PLI --> PLG

    style CLI fill:#ff9999,stroke:#cc0000
    style DB fill:#99ccff,stroke:#0066cc
    style EMB fill:#99ffcc,stroke:#00cc66
    style CLU fill:#ffcc99,stroke:#cc6600
    style WEB fill:#cc99ff,stroke:#6600cc
    style RAG fill:#ffff99,stroke:#cccc00
    style MCP fill:#ff99cc,stroke:#cc0066
    style REG fill:#99ffff,stroke:#00cccc
```

## CLI Command Flow

The CLI (`cli/` package) is the primary entry point. All commands
follow a repeated pattern: parse args → load config → initialize resources → execute → handle errors.

```{mermaid}
graph TD
    MAIN["main()"] --> PARSE[Parse Arguments]
    PARSE --> DISPATCH{Command?}

    DISPATCH -->|download| DL[download_command]
    DISPATCH -->|create-embeddings| CE[create_embeddings_command]
    DISPATCH -->|pre-process| PP[pre_process_command]
    DISPATCH -->|search| SR[search_command]
    DISPATCH -->|chat| CH[chat_command]
    DISPATCH -->|web-ui| WU[web_ui_command]
    DISPATCH -->|clustering run| CL[cluster_embeddings_command]
    DISPATCH -->|clustering clear-cache| CC[clear_clustering_cache_command]
    DISPATCH -->|clustering pre-generate| PG[pre_generate_clustering_command]
    DISPATCH -->|delete-data| DD[delete_data_command]
    DISPATCH -->|mcp-server| MS[mcp_server_command]
    DISPATCH -->|eval generate| EG[eval_generate_command]
    DISPATCH -->|eval verify| EV[eval_verify_command]
    DISPATCH -->|eval run| ER[eval_run_command]
    DISPATCH -->|eval results| ERS[eval_results_command]
    DISPATCH -->|eval clear| ECL[eval_clear_command]
    DISPATCH -->|registry upload| RU[registry_upload_command]
    DISPATCH -->|registry download| RD[registry_download_command]
    DISPATCH -->|registry list| RL[registry_list_command]
    DISPATCH -->|registry delete| RDEL[registry_delete_command]

    subgraph "Repeated CLI Pattern (⚠ duplicated in each command)"
        A1[Load config] --> A2[Resolve conference/year]
        A2 --> A3[Print header banner]
        A3 --> A4[Validate embeddings DB]
        A4 --> A5[Init EmbeddingsManager]
        A5 --> A6[Test LM Studio connection]
        A6 --> A7[Init DatabaseManager]
        A7 --> A8[Execute command logic]
        A8 --> A9[Error handling + traceback]
    end

    DL --> A1
    CE --> A1
    SR --> A1
    CH --> A1
    CL --> A1
    PG --> A1

    style A1 fill:#ffcccc
    style A2 fill:#ffcccc
    style A3 fill:#ffcccc
    style A4 fill:#ffcccc
    style A5 fill:#ffcccc
    style A6 fill:#ffcccc
    style A7 fill:#ffcccc
    style A9 fill:#ffcccc
```

## Data Pipeline Flow

```{mermaid}
graph LR
    subgraph "Data Ingestion"
        API[Conference APIs<br/>OpenReview, etc.] -->|download| PLG[Plugin System]
        PLG -->|LightweightPaper| DB[(SQL Database<br/>SQLite/PostgreSQL)]
    end

    subgraph "Embedding Generation"
        DB -->|paper text| EMB[EmbeddingsManager]
        LLM1[LLM Backend<br/>LM Studio/Blablador] -->|embedding vectors| EMB
        EMB -->|store| CHR[(ChromaDB)]
    end

    subgraph "Analysis & Search"
        CHR -->|embeddings| CLU[ClusteringManager]
        CLU -->|cache results| DB
        CHR -->|similarity search| SEM[Semantic Search]
        DB -->|keyword search| KW[Keyword Search]
    end

    subgraph "User Interfaces"
        SEM --> WEB[Web UI]
        KW --> WEB
        CLU --> WEB
        SEM --> RAG[RAG Chat]
        CLU --> RAG
        RAG --> MCP[MCP Tools]
        WEB --> EXP[Export ZIP/Markdown]
    end

    subgraph "Registry"
        DB -->|export| REG[OCI Registry]
        CHR -->|export| REG
        REG -->|import| DB
        REG -->|import| CHR
    end
```

## Database Layer Flow

```{mermaid}
graph TD
    subgraph "DatabaseManager mixins (db/ package)"
        direction TB
        CONN["Connection Management<br/>__init__, connect, close, __enter__, __exit__"]

        subgraph "Paper Operations"
            ADD["add_paper / add_papers"]
            SEARCH["keyword_search<br/>(flexible filter builder)"]
            QUERY["search<br/>(query parser + router)"]
        end

        subgraph "⚠ Duplicate: Faceting Methods"
            SESS["get_sessions(conference, year)"]
            CONF["get_conferences(year)"]
            YEARS["get_years(conference)"]
            CY["get_conference_years_from_db()"]
        end

        subgraph "Clustering Cache"
            CGET["get_clustering_cache"]
            CSAVE["save_clustering_cache"]
            CDEL["delete_clustering_cache_by_conference_year"]
            CCNT["count_clustering_cache_by_conference_year"]
            CCLR["clear_clustering_cache"]
        end

        subgraph "⚠ Duplicate: CRUD Patterns"
            direction LR
            QA["Eval QA Pairs<br/>add, get, count, update, delete"]
            ER["Eval Results<br/>add, get, delete"]
        end

        subgraph "Import/Export"
            EXPS["export_papers_to_sqlite"]
            IMPS["import_papers_from_sqlite"]
            EXPC["export_clustering_cache_to_json"]
            IMPC["import_clustering_cache_from_json"]
        end
    end

    CONN --> ADD
    CONN --> ADD
    CONN --> SEARCH
    CONN --> QUERY
    SEARCH --> QUERY
    CONN --> SESS
    CONN --> CONF
    CONN --> YEARS
    CONN --> CGET
    CONN --> QA
    CONN --> ER
    CONN --> EXPS

```

## Embeddings & Clustering Flow

```{mermaid}
graph TD
    subgraph "EmbeddingsManager"
        ECONN["connect() → ChromaDB"]
        ECOL["create_collection()"]
        EADD["add_paper / embed_from_database<br/>→ generate_embedding(s) → store"]
        ESRCH["search_similar (low level)"]
        EQRY["_query_embeddings"]
        EFIND["DatabaseManager.find_paper_ids_within_distance<br/>(UIDs + distances)"]
        EQUERY["query_ids_within_distance<br/>(IDs + distances)"]
        EIMP["import/export_embeddings"]

        subgraph "SQL filter resolution"
            W1["DatabaseManager.search:<br/>SQL filters → UIDs → Chroma ids"]
            W2["_query_embeddings:<br/>IDs + distances only"]
        end
    end

    subgraph "ClusteringManager"
        CLOAD["load_embeddings"]
        CREDUC["reduce_dimensions<br/>(t-SNE)"]
        CCLUST["cluster<br/>(Agglomerative, ward linkage)"]
        CKEYS["extract_cluster_keywords"]
        CLBL["generate_cluster_labels"]
        CHRCH["generate_hierarchical_labels"]
        CRES["get_clustering_results"]

        subgraph "Shared keyword extraction"
            T1["extract_cluster_keywords"]
            T2["_extract_keywords_for_samples"]
            T3["compute_tfidf_keywords<br/>(shared TF-IDF implementation)"]
        end

        subgraph "Shared LLM label generation"
            L1["_generate_llm_label"]
            L2["_generate_parent_label_llm"]
            L3["_generate_llm_label_from_keywords"]
            L4["_call_llm_for_label<br/>(shared call + fallback)"]
        end

        subgraph "SQL filter resolution"
            W3["load_embeddings:<br/>resolve UIDs in SQL"]
        end
    end

    subgraph "Standalone Functions"
        PF["perform_clustering<br/>(convenience wrapper)"]
        CWC["compute_clusters_with_cache<br/>(multi-level caching)"]
        W4["resolves conference/year filters to UIDs"]
    end

    ECONN --> ECOL --> EADD
    ECOL --> ESRCH
    ECOL --> EQRY
    EQRY --> EFIND
    EQRY --> EQUERY
    CLOAD --> CREDUC --> CCLUST --> CKEYS --> CLBL
    CCLUST --> CHRCH
    CLBL --> CRES

    PF --> CLOAD
    CWC --> CLOAD


```

## RAG Chat & MCP Tools Flow

```{mermaid}
graph TD
    subgraph "RAG Chat (rag/ package)"
        RQUERY["query()"]
        RBUILD["_build_agent()"]

        subgraph "Explicit Tool Wrappers (8)"
            TW1["_tool_search_papers"]
            TW2["_tool_get_conference_topics"]
            TW3["_tool_get_topic_evolution"]
            TW4["_tool_analyze_topic_relevance"]
            TW5["_tool_get_cluster_visualization"]
            TW6["_tool_get_paper_details"]
            TW7["_tool_summarize_interesting_papers"]
            TW8["_tool_recommend_related_papers"]
        end

        RH["_run_rag_tool<br/>(logging + result capture + formatting)"]
    end

    subgraph "MCP Tools (mcp_tools.py)"
        EXEC["execute_mcp_tool<br/>(dispatcher)"]
        NORM["_normalize_args(arguments, rules)"]
        RULES["NormRule rule tables"]
        FORMAT["format_tool_result_for_llm"]
    end

    subgraph "MCP Server (mcp/ package)"
        subgraph "Core Tool Functions"
            S1["search_papers"]
            S2["get_conference_topics"]
            S3["get_topic_evolution"]
            S4["analyze_topic_relevance"]
            S5["get_cluster_visualization"]
            S6["get_paper_details"]
            S7["summarize_interesting_papers"]
            S8["recommend_related_papers"]
        end

        RES["_get_mcp_resources()<br/>_get_clustering_resources()"]

    end

        TW1 & TW2 & TW3 & TW4 & TW5 & TW6 & TW7 & TW8 --> RH

    TW1 --> S1
    TW2 --> S2
    TW3 --> S3
    TW4 --> S4
    TW5 --> S5
    TW6 --> S6
    TW7 --> S7
    TW8 --> S8

    EXEC --> NORM
    NORM --> RULES
    EXEC --> S1 & S2 & S3 & S4 & S5 & S6 & S7 & S8
    EXEC --> FORMAT

    S1 --> WB
    S1 & S3 & S4 & S8 --> RES
    S6 & S7 --> RES
    S2 & S5 --> RES
```

## Web UI Request Flow

```{mermaid}
graph TD
    subgraph "Flask Web UI (app.py - 19 routes)"
        REQ[HTTP Request] --> MW["Middleware<br/>ProxyFix, CORS, teardown_db"]

        MW --> IDX["GET / → index()"]
        MW --> CIDX["GET /<conference> → conference_index()"]
        MW --> HLTH["GET /health"]
        MW --> STAT["GET /api/stats"]
        MW --> EMOD["GET /api/embedding-model-check"]
        MW --> FILT["GET /api/filters"]
        MW --> AFILT["GET /api/available-filters"]
        MW --> SRCH["POST /api/search"]
        MW --> GPAP["GET /api/paper/<uid>"]
        MW --> BATCH["POST /api/papers/batch"]
        MW --> CHAT["POST /api/chat"]
        MW --> RSET["POST /api/chat/reset"]
        MW --> CCMP["POST /api/clusters/compute"]
        MW --> CSRCH["POST /api/clusters/search"]
        MW --> EXPRT["POST /api/export/interesting-papers"]
        MW --> DOND["POST /api/donate-data"]
        MW --> DONC["POST /api/donate-chat"]

        subgraph "⚠ Repeated Pattern in All Routes"
            P1["db = get_database()"]
            P2["try/except + logger.error + jsonify"]
            P3["Parameter validation + type conversion"]
        end
    end

    SRCH -->|keyword| DB[(DatabaseManager)]
    SRCH -->|semantic| EMB[(EmbeddingsManager)]
    CHAT --> RAG[RAGChat]
    CCMP --> CLU[ClusteringManager]
    EXPRT --> EXU[export_utils]
    GPAP --> PAP[paper_utils]

    style P1 fill:#ffcccc,stroke:#cc0000
    style P2 fill:#ffcccc,stroke:#cc0000
    style P3 fill:#ffcccc,stroke:#cc0000
```

## Export & Paper Utilities Flow

```{mermaid}
graph TD
    subgraph "export/ package"
        EXP["export_papers_to_zip"] --> GFS["generate_folder_structure_export"]
        GFS --> GMR["generate_main_readme"]
        GFS --> GAP["generate_all_papers_markdown"]
        GFS --> GST["generate_search_term_markdown"]

        subgraph "⚠ Duplicate: Paper formatting"
            GAP_F["generate_all_papers_markdown:<br/>group by session → format paper block"]
            GST_F["generate_search_term_markdown:<br/>group by session → format paper block<br/>⚠ nearly identical to GAP"]
        end
    end

    subgraph "paper_utils.py"
        GPA["get_paper_with_authors"] --> DB[(DatabaseManager)]
        FSR["format_search_results"] --> GPA
    end

    style GAP_F fill:#ffcccc,stroke:#cc0000
    style GST_F fill:#ffcccc,stroke:#cc0000
```

## Registry Upload/Download Flow

```{mermaid}
graph LR
    subgraph "RegistryClient"
        UP["upload()"] --> EY["_export_year()"]
        EY --> PT["_push_tag()"]

        DL["download()"] --> FM["_find_best_matching_tag()"]
        FM --> IY["_import_year()"]
        IY --> CE["_check_embedding_model()"]

        subgraph "⚠ Duplicate: Progress callbacks"
            PC1["upload: def _progress(...)"]
            PC2["download: def _progress(...)"]
            PC3["_export_year: def _progress(...)"]
            PC4["_import_year: def _progress(...)"]
        end
    end

    EY --> DB[(DatabaseManager<br/>export_papers_to_sqlite<br/>export_clustering_cache_to_json)]
    EY --> EMB[(EmbeddingsManager<br/>export_embeddings)]

    IY --> DB2[(DatabaseManager<br/>import_papers_from_sqlite<br/>import_clustering_cache_from_json)]
    IY --> EMB2[(EmbeddingsManager<br/>import_embeddings)]

    style PC1 fill:#ffcccc,stroke:#cc0000
    style PC2 fill:#ffcccc,stroke:#cc0000
    style PC3 fill:#ffcccc,stroke:#cc0000
    style PC4 fill:#ffcccc,stroke:#cc0000
```

## Refactoring Status

The major duplication patterns identified during the refactor have been
resolved. The table records where each concern now lives.

| # | Pattern | Now |
|---|---------|-----|
| 1 | SQL filter/UID resolution | `db/search.py` (`resolve_paper_uids`, shared filter builder) |
| 2 | CLI header printing | `cli/_common.py` (`_print_header`) |
| 3 | TF-IDF keyword extraction | `clustering/tfidf.py` (fit once, shared) |
| 4 | LLM label generation | `clustering/labeling.py` + `llm_utils.py` |
| 5 | RAG tool execution/logging tail | `rag/tools.py` (`_run_rag_tool`) |
| 6 | MCP argument normalization | `mcp_tools.py` (`NormRule` + `_normalize_args`) |
| 7 | Resource init in MCP tools | `mcp/resources.py` (`_get_mcp_resources`) |
| 8 | Conference/year argument resolution | `cli/_common.py` (`_resolve_conference_arg`) |
| 9 | Paper markdown formatting | `export/markdown.py` |
| 10 | Error handling + traceback pattern | `cli/_common.py` (`_unexpected_error`) |
| 11 | Faceting query pattern | `db/catalog.py` |
| 12 | Eval CRUD parallel structures | `db/evaluation.py` |
| 13 | Progress callback definitions | `registry.py` |
| 14 | Web route error handling | `web_ui/api_utils.py` (`api_error_handler`) |
| 15 | Timestamps / JSON / URL / conference parsing | `utils/` package |
| 16 | Tool descriptions/examples (evaluation drift) | `mcp_tools.py` (`TOOL_DESCRIPTIONS`) |
