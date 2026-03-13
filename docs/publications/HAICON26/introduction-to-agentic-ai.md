# Introduction to Agentic AI

This document provides an overview over possible topics for an introductory presentation for the Agentic AI Workshop that is supposed to provide important information on how Agentic Systems work, what key concepts are important and an overview over possible use cases and popular agentic solutions for research (i.e. [karpathy/autoresearch: AI agents running research on single-GPU nanochat training automatically](https://github.com/karpathy/autoresearch)).

### **1. Foundational Concepts**
* Definition of an agent vs a plain LLM
* Differences between autonomous, semi-autonomous, and human-in-the-loop agents
* Planner–executor loop
* Core components of agents:
    * Memory (short-term, long-term, episodic)
    * Action interfaces (tool calling, APIs)
    * Feedback loops

### **2. Historical Context**
* Why did it take so long for Agents to emerge
* What was enabling usage of LLMs as agents

### **3. Mechanisms of Agentic Workflows**
* Tool calling and integration
    * Dynamic tool discovery
    * MCP servers (Multi-Client Platform / Middleware Control Plane)
    * Permissions (What can go wrong?)
* Planning and scheduling concepts in agents
    * Cron jobs and periodic actions
    * Event-driven triggers and automation loops

### **4. Overview over Agentic Tools and Platforms**
* LangChain, OpenClaw, n8n, Claude Code & Claude Cowork, GitHub CLI + GitHub Issues agents, ...
* Comparison criteria (TODO: Structure the tools regarding shared properties/ use cases/ ...)
* Overview over popular projects that use agentic workflows to support research
  * [karpathy/autoresearch: AI agents running research on single-GPU nanochat training automatically](https://github.com/karpathy/autoresearch)
  * TODO: collect projects to reference here

### **5. Concepts Within Agentic Systems**
* Skills / Plugins
* Roles / Personas / Expert profiles
* Embedding management for retrieval and reasoning

### **6. Practical Research Use Cases**
* Reference https://hal.science/hal-05463006
* Provide list only without going into detail
    * Literature search and summarization agents
    * Abstract exploration and paper mapping agents
    * Idea generation, testing, and prototyping
    * Experimental design automation
    * Dataset curation and preprocessing
    * Citation network analysis and automated reviews
    * Meeting and research note summarization
    * Workflow orchestration across multiple tools

### **7. Current Issues & Challenges**
* Context window limitations and strategies to overcome
* Security risks and sandboxing concerns
* Outdated knowledge / embedding staleness
* Hallucinations and error propagation in multi-step reasoning
* Multi-agent coordination complexity
* Explainability and auditability of agent actions
* Cost and computational resource considerations
* Ethical concerns: bias amplification, privacy, and accountability

### **8. Advancements and Emerging Features**
* MCP Apps (modular agentic applications)
* Sandboxing and secure execution environments
* Advanced memory structures (vector DBs, retrieval-based agents)
* Replacement of MCP with command line access
* Agent orchestration frameworks for research teams
* Incremental learning in deployed agents
* Agents running and performing Research 24/7 (see OpenFang?)