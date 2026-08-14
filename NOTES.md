# Implementation Notes

All work is in `src/pipeline.py`. `src/knowledge_base.py` is unmodified.

## TODO 1: `ask_question()`

Retrieves the top 3 chunks with `similarity_search(question, k=3)`, joins their
text into a context string, fills `PROMPT_TEMPLATE`, calls the LLM, and returns
`{"answer": str, "sources": list[str]}`.

**One deliberate addition: the context is capped at 1,200 characters.**
`get_llm()` truncates its input at 512 tokens, and the question sits at the *end*
of the prompt. Three 500-character chunks plus the template overflow that window,
so the question is the first thing cut, and the model then answers a question it
never actually saw. `_build_context()` fills greedily from the best match down,
which keeps the highest-scoring chunk intact and drops only the weakest text.
All three retrieved chunks are still returned in `"sources"` regardless.

## TODO 2: `main()`

Builds the knowledge base, loads the LLM, then loops on `input()`. Exits on
`quit` / `exit` / `q`, and on Ctrl-C or EOF.

## Bonus items

- **`--query` flag** for single-question mode (`python -m src.pipeline --query "..."`).
- **Error handling.** Blank input re-prompts, a missing `data/` directory exits
  with a clear message, setup failures are reported instead of tracebacking, and
  a generation error inside the loop doesn't kill the session.
- **Type hints** throughout.
- **Extra tests** in `tests/test_pipeline_offline.py`: 15 unit tests using stub
  doubles for the vector store and LLM. They run in seconds with no model
  downloads and cover what a model-backed test can't pin down deterministically:
  that `k=3` is actually requested, that the prompt contains both context and
  question, that the question survives an oversized context, context-budget
  edge cases, empty input, empty retrieval, and blank generations.

## An observation on the `data/` directory

`data/` contains two files that aren't in the README's project structure:
`company_handbook.txt` and `product_faq.txt`, both about "Acme Corp" / AcmeCloud
rather than the marketing agency. The pre-built loader globs `**/*.txt`, so they
are indexed. **11 of 38 chunks (29%) are unrelated to the agency**, and
AcmeCloud's own "plans / pricing / Enterprise / support" vocabulary competes
directly with the agency's pricing and FAQ content. In practice a question like
"Can I cancel early?" can spend retrieval slots on AcmeCloud text.

Two more retrieval hazards worth flagging:

- Chunking separates some prices from their descriptions. The Growth package
  price line exists as a standalone 29-character chunk, and the Enterprise
  package price is detached from the paragraph it prices.
- The agency is named "**Starter** marketing agency", so the token "Starter"
  opens all three agency files, which makes "How much is the Starter package?"
  lexically ambiguous.

I left retrieval as the specified `k=3` rather than adding a score-gap filter or
over-fetch-and-rerank. Those heuristics need to be tuned against the real
MiniLM embedding distribution, and an untuned threshold can silently drop good
chunks, a worse failure than a diluted context. Noting it as a known limitation
with a clear fix path seemed better than shipping a guess.

## Verifying

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v          # downloads ~1.2GB of models on first run
```

`offline_check.py` runs the official suite with local stand-ins for the two
HuggingFace models (TF-IDF vectors, extractive generator) against the real data,
real chunking, and a real FAISS index. It is a development aid for running the
suite without network access to huggingface.co, not part of the assignment.
