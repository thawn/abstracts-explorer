# Evaluation Module

The evaluation module implements automatic evaluation of the RAG system using
an LLM-as-judge approach.

## Features

- Generate evaluation question-answer pairs from the paper database and MCP tools
- Run the RAG pipeline on stored Q/A pairs and score output with an LLM judge
- Compute summary statistics and format results for display
- Support for follow-up question evaluation

## Quick Start

```python
from abstracts_explorer.evaluation import Evaluator
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.database import DatabaseManager

em = EmbeddingsManager()
db = DatabaseManager()

evaluator = Evaluator(em, db)

# Generate Q/A pairs using the LLM
pairs = evaluator.generate_qa_pairs(n_pairs_per_tool=2)
evaluator.store_qa_pairs(pairs)

# Run evaluation
summary = evaluator.run_evaluation()
print(summary)
```

## API Reference

```{eval-rst}
.. automodule:: abstracts_explorer.evaluation
   :members:
   :undoc-members:
   :show-inheritance:
```
