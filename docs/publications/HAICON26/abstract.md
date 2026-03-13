# Abstracts Explorer: LLM-Powered Semantic Exploration of Scientific Conference Proceedings

## Authors

Till Korten, Peter Steinbach

## Abstract

Keeping pace with the rapidly growing machine learning literature is a major challenge. Conferences such as NeurIPS, ICLR, and ICML now accept thousands of papers annually, making it difficult for researchers to identify relevant work and track emerging trends. We present Abstracts Explorer, an open-source Python toolkit that combines LLM-based semantic search, unsupervised clustering, and retrieval-augmented generation (RAG) to enable intelligent exploration of conference proceedings at scale.

The system implements a modular pipeline with four components: (1) A plugin-based data acquisition framework that downloads and normalizes paper metadata from multiple conference APIs into a unified relational database (SQLite or PostgreSQL). New conferences can be added by implementing a lightweight downloader plugin. (2) A vector embedding engine backed by ChromaDB that generates dense representations of paper abstracts via any OpenAI-compatible embedding model, enabling sub-second semantic similarity search over tens of thousands of papers. (3) An unsupervised clustering and visualization module supporting five algorithms (K-Means, DBSCAN, Agglomerative, Spectral, Fuzzy C-Means) and three dimensionality reduction methods (PCA, t-SNE, UMAP), augmented by LLM-powered automatic cluster labeling. (4) A RAG chat interface enabling natural-language questions about the corpus, with automatic tool-use integration via the Model Context Protocol (MCP) for topic trend analysis and cluster exploration.

A key design choice is native MCP integration: the system exposes analytical capabilities — topic frequency, temporal trend detection, and research landscape visualization — as standardized MCP tools invocable by any compatible LLM client. The RAG module automatically routes queries to appropriate tools when questions concern topics or trends, transparently combining vector retrieval with structured cluster analysis.

The toolkit ships with a Flask web interface featuring semantic and keyword search, conversational exploration, paper bookmarking with multi-format export, and interactive Plotly-based cluster visualizations. Docker Compose enables single-command deployment of the full stack. We demonstrate the system on NeurIPS 2025 proceedings, showing how semantic clustering reveals thematic structure, RAG surfaces papers that keyword search misses, and MCP tool integration enables LLM agents to autonomously analyze research trends. Abstracts Explorer is available under the Apache 2.0 license at https://github.com/thawn/abstracts-explorer.

Keywords: scientific literature exploration, semantic search, retrieval-augmented generation, unsupervised clustering, large language models, Model Context Protocol