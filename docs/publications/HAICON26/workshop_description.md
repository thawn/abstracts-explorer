# Building Agentic ML Tools for Science: What Works and What Doesn't

## Workshop Description

Large language models and agentic AI systems are increasingly being adopted as tools in scientific workflows — from literature review and data analysis to experimental design and code generation. Yet building *effective* agentic tools for research remains as much craft as science: architectural choices around retrieval strategies, tool-use protocols, embedding models, and human-in-the-loop interfaces can make or break a system's practical utility. At the same time, the rapid pace of development means that hard-won lessons about what works (and what fails) are rarely shared systematically across projects.

This workshop brings together developers and users of agentic ML tools for science to openly exchange practical knowledge and experience. Rather than focusing on algorithmic novelty, we emphasize **engineering reality**: Which design patterns lead to reliable agentic behavior? Where do current LLM-based tools fall short in scientific contexts? How do we evaluate whether an agentic tool genuinely accelerates research or merely creates an illusion of productivity?

As a basis for discussion, we present concrete example projects that tackle different facets of agentic scientific tooling — from semantic literature exploration with retrieval-augmented generation and clustering MCP tools to knowledge-graph-based reasoning systems. By dissecting these systems in detail, including their failures and limitations, we aim to distill transferable principles for the community.

### Guiding Questions

- **Architecture & Integration:** How should agentic tools expose capabilities to LLMs (e.g., via the Model Context Protocol)? What are effective patterns for combining retrieval, structured analysis, and generation?
- **Evaluation & Trust:** How do we measure whether an agentic science tool is actually helpful? When should we trust LLM-generated analyses of scientific literature?
- **Practical Pitfalls:** What are common failure modes — hallucinated references, embedding model drift, brittle tool-use chains — and how can they be mitigated?
- **Human-in-the-Loop:** What level of autonomy is appropriate? Where must human oversight remain, and how do we design interfaces that support it?

### Example Projects

1. **Abstracts Explorer** — An open-source Python toolkit combining LLM-based semantic search, unsupervised clustering, and RAG to explore conference proceedings (NeurIPS, ICLR, ICML) at scale. Features native MCP integration for LLM-driven topic and trend analysis, a Flask web interface, and Docker-based deployment. ([github.com/thawn/abstracts-explorer](https://github.com/thawn/abstracts-explorer))

2. **Voucher Canvas Agent** — A prototype system for building and querying scientific knowledge graphs using LLMs. It integrates with existing graph databases and provides a simple API for querying and updating the knowledge base. ([github.com/thawn/voucher-canvas-agent](https://github.com/thawn/voucher-canvas-agent))

## Agenda

### Morning — Presentations — Architecture walk-through, live demo, design trade-offs, and lessons learned & Discussion (2 h)

| Time    | Session                                                                                                                                |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| ~30 min | **Introduction to Agentic AI for Science**                                                                                             |
| ~30 min | **Project Presentation: Abstracts Explorer**                                                                                           |
| ~30 min | **Project Presentation: Voucher Canvas Agent Reasoner**                                                                                |
| ~15 min | *Break*                                                                                                                                |
| ~30 min | **Project Presentation: TBD**                                                                                                          |
| ~15 min | **General Discussion** — Cross-cutting themes: What design patterns transferred? Where did things break? What would we do differently? |

### Afternoon — Hackathon (2 h)

Participants form small groups to **build or plan** agentic ML tools hands-on. Groups self-organize around topics proposed in the morning discussion. Possible tracks include:

- **Feature sprints** — Implement a concrete feature in one of the presented projects (e.g., a new MCP tool, a conference downloader plugin, an improved clustering pipeline). Bring a laptop and be ready to write code.
- **Project planning** — Sketch the architecture and roadmap for a new agentic tool addressing a participant's own research domain. Output: a brief design document or project proposal.
- **Integration experiments** — Wire up existing tools via MCP or other protocols and stress-test them with real scientific queries.

| Time    | Session                                                                                         |
| ------- | ----------------------------------------------------------------------------------------------- |
| ~15 min | **Group Formation** — Pitch topics, form groups (3–5 people).                                   |
| ~90 min | **Hands-on Work** — Small-group hacking / planning with roaming support from organizers.        |
| ~15 min | **Lightning Reports** — Each group presents what they built, learned, or planned (~2 min each). |

## Target Audience

Researchers, research software engineers, and data scientists who are building, evaluating, or considering agentic ML tools in their scientific workflows. No prior experience with agent frameworks is required — practical curiosity and a willingness to share experiences are the main prerequisites. For the afternoon hackathon, a laptop with Python 3.11+ and Git installed is recommended.

## Format

The workshop combines **presentation-driven discussion** in the morning with **hands-on collaborative work** in the afternoon. Morning sessions feature short project presentations interleaved with structured discussion. The afternoon hackathon lets participants turn ideas into code or concrete plans in small groups. Participants are encouraged to bring their own experiences, tools, and "war stories" to contribute throughout.

## Organizers

Till Korten is a Helmholtz AI Consultant and developer of Abstracts Explorer
Haider Khan is a Helmholtz AI Consultant and developer of Knowledge Graph Reasoner