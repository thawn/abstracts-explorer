# Codebase Architecture & Flow-Charts

This document provides a comprehensive architectural overview of the Abstracts Explorer
codebase, including module relationships, data flow, and function-level flow-charts.

## Module Overview

The codebase comprises **~21,800 lines** of Python across 20 source files:

| Module | Lines | Responsibility |
|--------|------:|----------------|
| `cli.py` | 3,446 | Command-line interface (25 commands) |
| `database.py` | 2,943 | SQL database operations (SQLAlchemy ORM) |
| `clustering.py` | 2,094 | Clustering algorithms & visualization |
| `registry.py` | 1,664 | OCI registry upload/download |
| `embeddings.py` | 1,372 | ChromaDB vector embeddings |
| `mcp_server.py` | 1,328 | MCP server & tool implementations |
| `web_ui/app.py` | 1,120 | Flask web API (22 routes) |
| `plugin.py` | 976 | Plugin system & data models |
| `rag.py` | 874 | RAG chat with Pydantic AI |
| `evaluation.py` | 833 | RAG evaluation framework |
| `mcp_tools.py` | 824 | MCP tool schema & formatting |
| `export_utils.py` | 675 | Paper export (ZIP/Markdown) |
| `config.py` | 532 | Configuration management |
| `db_models.py` | 435 | SQLAlchemy ORM models |
| `paper_utils.py` | 247 | Paper formatting utilities |
| `plugins/*.py` | ~2,177 | Conference downloader plugins (7 plugins) |

## High-Level Module Dependency Graph

```{mermaid}
graph TD
    CLI[cli.py<br/>3446 lines<br/>25 commands]
    CFG[config.py<br/>532 lines]
    DB[database.py<br/>2943 lines]
    DBM[db_models.py<br/>435 lines]
    EMB[embeddings.py<br/>1372 lines]
    CLU[clustering.py<br/>2094 lines]
    RAG[rag.py<br/>874 lines]
    MCP[mcp_server.py<br/>1328 lines]
    MCT[mcp_tools.py<br/>824 lines]
    WEB[web_ui/app.py<br/>1120 lines]
    PLG[plugin.py<br/>976 lines]
    PLI[plugins/*<br/>7 downloaders]
    EXP[export_utils.py<br/>675 lines]
    PAP[paper_utils.py<br/>247 lines]
    REG[registry.py<br/>1664 lines]
    EVL[evaluation.py<br/>833 lines]

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

The CLI (`cli.py`) is the primary entry point with 25 command functions. All commands
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
    subgraph "DatabaseManager - 2943 lines"
        direction TB
        CONN["Connection Management<br/>__init__, connect, close, __enter__, __exit__"]

        subgraph "⚠ Duplicate: Session Validation"
            SV1["if not self._session: raise DatabaseError<br/>(repeated in 30+ methods)"]
        end

        subgraph "Paper Operations"
            ADD["add_paper / add_papers"]
            SEARCH["search_papers<br/>(flexible filter builder)"]
            KWSEARCH["search_papers_keyword<br/>(⚠ thin wrapper around search_papers)"]
            AUTHSEARCH["search_authors_in_papers"]
        end

        subgraph "⚠ Duplicate: Faceting Methods"
            SESS["get_sessions(conference, year)"]
            CONF["get_conferences(year)"]
            YEARS["get_years(conference)"]
            YFC["get_years_for_conference(conference)<br/>⚠ duplicate of get_years"]
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

    CONN --> SV1
    SV1 --> ADD
    SV1 --> SEARCH
    SEARCH --> KWSEARCH
    SV1 --> SESS
    SV1 --> CONF
    SV1 --> YEARS
    YEARS -.->|"duplicate"| YFC
    SV1 --> CGET
    SV1 --> QA
    SV1 --> ER
    SV1 --> EXPS

    style YFC fill:#ffcccc,stroke:#cc0000
    style SV1 fill:#ffcccc,stroke:#cc0000
    style KWSEARCH fill:#ffffcc,stroke:#cccc00
```

## Embeddings & Clustering Flow

```{mermaid}
graph TD
    subgraph "EmbeddingsManager"
        ECONN["connect() → ChromaDB"]
        ECOL["create_collection()"]
        EADD["add_paper → generate_embedding → store"]
        ESRCH["search_similar / search_papers_semantic"]
        EFIND["find_papers_within_distance"]
        EIMP["import/export_embeddings"]

        subgraph "⚠ Duplicate: Where-clause building"
            W1["search_papers_semantic:<br/>build $and/$in filter"]
            W2["find_papers_within_distance:<br/>build $and/$in filter"]
        end
    end

    subgraph "ClusteringManager"
        CLOAD["load_embeddings"]
        CREDUC["reduce_dimensions<br/>(PCA / t-SNE / UMAP)"]
        CCLUST["cluster<br/>(K-Means / DBSCAN / Agglom / Spectral / Fuzzy)"]
        CKEYS["extract_cluster_keywords"]
        CLBL["generate_cluster_labels"]
        CHRCH["generate_hierarchical_labels"]
        CRES["get_clustering_results"]

        subgraph "⚠ Duplicate: TF-IDF extraction"
            T1["extract_cluster_keywords"]
            T2["_extract_keywords_for_samples<br/>⚠ nearly identical to T1"]
        end

        subgraph "⚠ Duplicate: LLM label generation"
            L1["_generate_llm_label"]
            L2["_generate_parent_label_llm<br/>⚠ same pattern as L1"]
            L3["_generate_llm_label_from_keywords<br/>⚠ same pattern as L1"]
        end

        subgraph "⚠ Duplicate: Where-clause building"
            W3["load_embeddings:<br/>build $and/$in filter"]
        end
    end

    subgraph "Standalone Functions"
        PF["perform_clustering<br/>(convenience wrapper)"]
        CWC["compute_clusters_with_cache<br/>(multi-level caching)"]
        W4["⚠ also builds where-clauses"]
    end

    ECONN --> ECOL --> EADD
    ECOL --> ESRCH
    ECOL --> EFIND
    CLOAD --> CREDUC --> CCLUST --> CKEYS --> CLBL
    CCLUST --> CHRCH
    CLBL --> CRES

    PF --> CLOAD
    CWC --> CLOAD

    W1 -.->|"duplicate logic"| W2
    W1 -.->|"duplicate logic"| W3
    W1 -.->|"duplicate logic"| W4
    T1 -.->|"duplicate logic"| T2
    L1 -.->|"duplicate logic"| L2
    L1 -.->|"duplicate logic"| L3

    style W1 fill:#ffcccc,stroke:#cc0000
    style W2 fill:#ffcccc,stroke:#cc0000
    style W3 fill:#ffcccc,stroke:#cc0000
    style W4 fill:#ffcccc,stroke:#cc0000
    style T1 fill:#ffcccc,stroke:#cc0000
    style T2 fill:#ffcccc,stroke:#cc0000
    style L1 fill:#ffcccc,stroke:#cc0000
    style L2 fill:#ffcccc,stroke:#cc0000
    style L3 fill:#ffcccc,stroke:#cc0000
```

## RAG Chat & MCP Tools Flow

```{mermaid}
graph TD
    subgraph "RAG Chat (rag.py)"
        RQUERY["query()"]
        RCHAT["chat()"]
        RBUILD["_build_agent()"]

        subgraph "⚠ Duplicate: Tool wrappers (6 identical patterns)"
            TW1["_tool_search_papers"]
            TW2["_tool_get_conference_topics"]
            TW3["_tool_get_topic_evolution"]
            TW4["_tool_analyze_topic_relevance"]
            TW5["_tool_get_cluster_visualization"]
            TW6["_tool_get_paper_details"]
        end
    end

    subgraph "MCP Tools (mcp_tools.py)"
        EXEC["execute_mcp_tool<br/>(dispatcher)"]
        SCHEMA["get_mcp_tools_schema"]

        subgraph "⚠ Duplicate: Normalization (4 similar functions)"
            N1["_normalize_search_papers_args"]
            N2["_normalize_get_topic_evolution_args"]
            N3["_normalize_analyze_topic_relevance_args"]
            N4["_normalize_get_paper_details_args"]
        end

        subgraph "⚠ Duplicate: Formatters (6 similar functions)"
            F1["_format_topic_relevance_result"]
            F2["_format_conference_topics_result"]
            F3["_format_topic_evolution_result"]
            F4["_format_search_papers_result"]
            F5["_format_visualization_result"]
            F6["_format_paper_details_result"]
        end
    end

    subgraph "MCP Server (mcp_server.py)"
        subgraph "Core Tool Functions"
            S1["search_papers"]
            S2["get_conference_topics"]
            S3["get_topic_evolution"]
            S4["analyze_topic_relevance"]
            S5["get_cluster_visualization"]
            S6["get_paper_details"]
        end

        subgraph "⚠ Duplicate: Resource init"
            RI["get_config → EmbeddingsManager → DatabaseManager<br/>(repeated in each tool function)"]
        end

        subgraph "⚠ Duplicate: Where-clause merging"
            WM1["merge_where_clause_with_conference"]
            WM2["merge_where_clause_with_years"]
        end
    end

    RBUILD --> TW1 & TW2 & TW3 & TW4 & TW5 & TW6

    TW1 --> S1
    TW2 --> S2
    TW3 --> S3
    TW4 --> S4
    TW5 --> S5
    TW6 --> S6

    EXEC --> N1 --> S1
    EXEC --> N2 --> S3
    EXEC --> N3 --> S4
    EXEC --> N4 --> S6

    S1 --> WM1
    S1 --> WM2

    style TW1 fill:#ffcccc,stroke:#cc0000
    style TW2 fill:#ffcccc,stroke:#cc0000
    style TW3 fill:#ffcccc,stroke:#cc0000
    style TW4 fill:#ffcccc,stroke:#cc0000
    style TW5 fill:#ffcccc,stroke:#cc0000
    style TW6 fill:#ffcccc,stroke:#cc0000
    style N1 fill:#ffcccc,stroke:#cc0000
    style N2 fill:#ffcccc,stroke:#cc0000
    style N3 fill:#ffcccc,stroke:#cc0000
    style N4 fill:#ffcccc,stroke:#cc0000
    style F1 fill:#ffffcc,stroke:#cccc00
    style F2 fill:#ffffcc,stroke:#cccc00
    style F3 fill:#ffffcc,stroke:#cccc00
    style F4 fill:#ffffcc,stroke:#cccc00
    style F5 fill:#ffffcc,stroke:#cccc00
    style F6 fill:#ffffcc,stroke:#cccc00
    style RI fill:#ffcccc,stroke:#cc0000
```

## Web UI Request Flow

```{mermaid}
graph TD
    subgraph "Flask Web UI (app.py - 22 routes)"
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
        MW --> CCCH["GET /api/clusters/cached"]
        MW --> CDEF["GET /api/clusters/default-count"]
        MW --> CSRCH["POST /api/clusters/search"]
        MW --> GYRS["GET /api/years"]
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
    subgraph "export_utils.py"
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
        BCP["build_context_from_papers"]
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

## Summary of Duplicate Code Paths

The following table summarizes all identified duplicate code patterns, ordered by
severity:

| # | Pattern | Occurrences | Modules | Severity |
|---|---------|-------------|---------|----------|
| 1 | Session validation boilerplate (`if not self._session`) | 30+ | database.py | High |
| 2 | ChromaDB where-clause construction (`$and/$in` filters) | 5+ | embeddings.py, clustering.py, mcp_server.py | High |
| 3 | EmbeddingsManager initialization sequence | 6 | cli.py | High |
| 4 | LM Studio connection test + error message | 4 | cli.py | Medium |
| 5 | Embeddings DB path validation | 4 | cli.py | Medium |
| 6 | CLI command header printing | 15+ | cli.py | Medium |
| 7 | TF-IDF keyword extraction | 2 | clustering.py | Medium |
| 8 | LLM label generation (OpenAI call + fallback) | 3 | clustering.py | Medium |
| 9 | RAG tool wrapper functions (identical pattern) | 6 | rag.py | Medium |
| 10 | MCP argument normalization functions | 4 | mcp_tools.py | Medium |
| 11 | Resource init in MCP tools (config → embed → db) | 6 | mcp_server.py | Medium |
| 12 | Conference/year argument resolution | 8+ | cli.py | Low |
| 13 | Paper markdown formatting (session grouping) | 2 | export_utils.py | Low |
| 14 | Error handling + traceback pattern | 15+ | cli.py | Low |
| 15 | Faceting query pattern | 4 | database.py | Low |
| 16 | Eval CRUD parallel structures | 2×5 | database.py | Low |
| 17 | Progress callback definitions | 4 | registry.py | Low |
| 18 | Web route error handling (try/except/jsonify) | 22 | web_ui/app.py | Low |
| 19 | `get_years_for_conference` duplicates `get_years` | 2 | database.py | Low |
