# tryRAG
A small exploration of retrieval-augmented generation (RAG) in Python.

## Data Design

* **Input File Format**: The system uses JSON Lines format, where each line represents a document and must include the `text` and `url` fields.
* **Chunking Strategy**: The `chunk_level` can be configured in `framework.py` to determine the granularity of splitting:

  * `web_page`: Treats the entire webpage content as a single chunk.
  * `paragraph`: Splits by paragraphs, with the option to retain surrounding context (`more_info`).
  * `sentence`: Further splits paragraphs into individual sentences.

After building the index, it can be saved as either a FAISS file (dense vector) or a JSON file (sparse BM25), enabling quick reloading in future runs.

## Retrieval Modes

The framework supports three retrieval modes, specified by the `mode` parameter in the configuration file:

1. **dense**: Uses `SentenceTransformer` to generate embeddings and performs FAISS-based nearest neighbor search.
2. **sparse**: Utilizes built-in BM25 for keyword matching.
3. **hybrid**: Combines both dense and sparse mechanisms, merging results and removing duplicates.

## Additional Enhancements

* **`use_upper_text`**: When set to `True` in the `ask()` function, the model uses the upper-level full text corresponding to the chunk to provide richer context.
* **`use_pre_answer`**: When set to `True` in the `ask()` first generates a predicted answer based on the query, then feeds both the prediction and retrieved documents into the model to enhance coverage.
* **`more_info`**: In paragraph mode, adjacent paragraphs can be merged into the chunk to preserve more contextual information.
* **Index Save/Load Support**: `save_index()` and `load_index()` allow saving and loading vector or BM25 parameters along with document content from a specified path, facilitating repeated experiments.
* **Batch Testing & Evaluation**: `experiment.py`, in combination with `batchTestRunner` and `Evaluator` from `evaluate.py`, supports batch execution of multiple configurations, computing BLEU, ROUGE, BERTScore, and hit rate metrics.

## Repository Layout

- `tryRAG/framework.py` – main class with indexing and generation logic.
- `tryRAG/dataType.py` – dataclasses for document chunks and query objects.
- `tryRAG/templates.py` – prompt templates used by the framework.
- `tryRAG/evaluate.py` – evaluate and data loading methods.
- `experiment.py` – run batch testing of experiments.
- `demo.ipynb` – Jupyter notebook that demonstrates basic usage.

## Installation
1. Create a Python environment (tested with Python 3.10):
   ```bash
   conda create --name tryRAG python=3.10
   conda activate tryRAG
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/yasaisen/tryRAG.git
   ```
3. Install the dependencies:
   ```bash
   pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
   pip install transformers==4.51.3 accelerate==0.26.0 faiss-cpu==1.11.0.post1 sentence-transformers==5.0.0 evaluate==0.4.5
   ```

## Usage

1. Prepare a JSON lines file where each line contains a document with `text` and `url` fields.
2. Create the framework and load your documents:

```python
from tryRAG.framework import RAGFramework

cfg = {
    "lm_model_name": "google/gemma-3-4b-it",
    "emb_model_name": "all-MiniLM-L6-v2",
    "mode": "hybrid", #@ "dense" / "sparse" / "hybrid"
    "chunk_level": "paragraph", #@ "web_page" / "paragraph" / "sentence"
    "more_info": True, #@ True / False
    "doc_path": "path/to/data.jsonl", 
    # "idx_path": "path/to/saved/dir",
    "device": "cuda",
}
USE_UPPER_TEXT = False
USE_PRE_ANSWER = False
TOP_K = 5

rag = RAGFramework.from_config(cfg)

```

3. Ask a question:

```python
question = 'Who is Lee Julian Purnell'

response = self.rag.ask(
   question, 
   top_k=TOP_K, 
   use_upper_text=USE_UPPER_TEXT, 
   pre_answer=USE_PRE_ANSWER, 
)
print(response["response"])
```

4. Run batch testing:

```python
from tryRAG.evaluate import Evaluator, batchTestRunner

testrunner = batchTestRunner(
    rag=rag, 
    dataset_type='optionalQA', #@ 'factoidQA' / 'optionalQA'
)

result_dict = testrunner.test(
   top_k=TOP_K, 
   use_upper_text=USE_UPPER_TEXT, 
   pre_answer=USE_PRE_ANSWER, 
)

evaluator = Evaluator()
eval_res = evaluator.evaluate(
    result_dict=result_dict,
)
print(eval_res)
```

The framework retrieves the most relevant document chunks, builds a prompt, and
uses a language model to produce an answer.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
