"""
Document Q&A Pipeline — YOUR WORK GOES HERE.

The knowledge base (loading, chunking, vector store) is already built
for you in knowledge_base.py. Your job is to:

  1. Retrieve relevant chunks and generate an answer
  2. Wire it up into an interactive CLI

Useful docs:
  - Vector store search: https://python.langchain.com/docs/how_to/vectorstores/
  - HuggingFace pipelines: https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/
"""

import argparse
import os
import sys
from typing import Callable, Dict, List

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.knowledge_base import build_knowledge_base

# Number of chunks pulled from the vector store for each question.
TOP_K = 3

# flan-t5-base truncates its input at 512 tokens. Three 500-character chunks
# plus the template can exceed that, and because the question sits at the END
# of the prompt it is the first thing to get cut, which produces answers that
# ignore what was actually asked. Capping the context keeps the question inside
# the window.
MAX_CONTEXT_CHARS = 1200

# Shown when retrieval or generation comes back empty.
NO_ANSWER = "I don't have enough information to answer that."


# ──────────────────────────────────────────────
# Provided: local LLM (no API key needed)
# ──────────────────────────────────────────────
def get_llm():
    """Return a callable local LLM using flan-t5-base.

    Downloads ~1GB on first run, then cached.
    Usage:
        llm = get_llm()
        result = llm("What color is the sky?")
        print(result[0]["generated_text"])  # "blue"
    """
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def generate(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=150)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": text}]

    return generate


# ──────────────────────────────────────────────
# Provided: prompt template
# ──────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful assistant for a marketing agency. Use the following context to answer the client's question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Client question: {question}

Answer:"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 1: Implement ask_question
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _build_context(sources: List[str], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Join retrieved chunks into one context string, bounded by max_chars.

    Chunks arrive best-match-first, so filling greedily from the front keeps the
    most relevant text and drops only the weakest matches when space runs out.
    """
    context, used = [], 0
    for source in sources:
        text = source.strip()
        if not text:
            continue
        remaining = max_chars - used
        if remaining <= 0:
            break
        context.append(text[:remaining])
        used += min(len(text), remaining) + 2  # +2 for the joining newlines
    return "\n\n".join(context)


def ask_question(vector_store, llm: Callable, question: str) -> Dict[str, object]:
    """Retrieve relevant chunks and generate an answer.

    Args:
        vector_store: FAISS vector store from knowledge_base.py
        llm: Callable from get_llm()
        question: The user's question string

    Returns:
        dict with two keys:
            "answer"  -> str: the generated answer
            "sources" -> list[str]: the chunk texts that were retrieved
    """
    question = (question or "").strip()
    if not question:
        return {"answer": "Please ask a question.", "sources": []}

    # 1. Retrieve the most relevant chunks.
    docs = vector_store.similarity_search(question, k=TOP_K)
    sources: List[str] = [doc.page_content for doc in docs]
    if not sources:
        return {"answer": NO_ANSWER, "sources": []}

    # 2 & 3. Combine the chunks into context and fill the prompt template.
    prompt = PROMPT_TEMPLATE.format(
        context=_build_context(sources), question=question
    )

    # 4. Generate, and extract just the answer text.
    result = llm(prompt)
    answer = result[0]["generated_text"].strip()

    return {"answer": answer or NO_ANSWER, "sources": sources}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 2: Complete the interactive loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def format_result(result: Dict[str, object]) -> str:
    """Render an ask_question() result for the terminal."""
    lines = ["", "📄 Sources:"]
    sources = result.get("sources") or []
    if sources:
        for i, source in enumerate(sources, start=1):
            snippet = " ".join(source.split())
            if len(snippet) > 200:
                snippet = snippet[:200].rstrip() + "..."
            lines.append(f"{i}. {snippet}")
    else:
        lines.append("(none)")
    lines.append("")
    lines.append(f"💬 Answer: {result.get('answer', NO_ANSWER)}")
    return "\n".join(lines)


def main() -> int:
    """Interactive Q&A loop (or a single answer with --query)."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    parser = argparse.ArgumentParser(
        description="Ask questions about the agency's services, pricing, and process."
    )
    parser.add_argument(
        "--query",
        help="Answer a single question and exit instead of starting the CLI loop.",
    )
    args = parser.parse_args()

    if not os.path.isdir(data_dir):
        print(f"Error: data directory not found at {os.path.abspath(data_dir)}")
        return 1

    # 1 & 2. Build the knowledge base and load the model.
    try:
        vector_store = build_knowledge_base(data_dir)
        print("Loading the language model (first run downloads ~1GB)...")
        llm = get_llm()
        print("  Done!\n")
    except Exception as exc:  # noqa: BLE001 - surface setup failures to the user
        print(f"Error: could not start the assistant ({exc})")
        return 1

    # Single-question mode.
    if args.query:
        print(format_result(ask_question(vector_store, llm, args.query)))
        return 0

    # 3. Interactive loop.
    print("Ask about our services, pricing, or process. Type 'quit' to exit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return 0

        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            return 0
        if not question:
            print("Please enter a question, or type 'quit' to exit.")
            continue

        try:
            print(format_result(ask_question(vector_store, llm, question)))
        except Exception as exc:  # noqa: BLE001 - keep the session alive on errors
            print(f"Sorry, something went wrong answering that ({exc}). Try again.")


if __name__ == "__main__":
    sys.exit(main())
