"""
Run the official test suite (tests/test_pipeline.py) without network access.

huggingface.co is unreachable from this sandbox, so the two downloaded models
are replaced with local stand-ins:
  * embeddings -> TF-IDF vectors (real FAISS index, real data, real chunking)
  * flan-t5    -> extractive stub that picks the best sentence from the context

Everything else (knowledge_base.py, pipeline.py, the tests) is untouched.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re
import numpy as np
import pytest
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer

import src.knowledge_base as kb
import src.pipeline as pipeline


class TfidfEmbeddings(Embeddings):
    """Deterministic local stand-in for all-MiniLM-L6-v2."""

    def __init__(self):
        self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)

    def embed_documents(self, texts):
        return self.vec.fit_transform(texts).toarray().astype("float32").tolist()

    def embed_query(self, text):
        return self.vec.transform([text]).toarray().astype("float32")[0].tolist()


def extractive_llm():
    """Stand-in for flan-t5-base: answers from the context, never echoes the prompt."""

    def generate(prompt):
        context = prompt.split("Context:", 1)[1].split("Client question:", 1)[0]
        question = prompt.split("Client question:", 1)[1].split("Answer:", 1)[0].strip()
        q_words = {w for w in re.findall(r"[a-z0-9,$]+", question.lower()) if len(w) > 2}
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", context) if s.strip()]
        if not sentences:
            return [{"generated_text": pipeline.NO_ANSWER}]
        best = max(
            sentences,
            key=lambda s: len(q_words & set(re.findall(r"[a-z0-9,$]+", s.lower()))),
        )
        return [{"generated_text": best[:300]}]

    return generate


kb.HuggingFaceEmbeddings = lambda *a, **k: TfidfEmbeddings()
pipeline.get_llm = extractive_llm

sys.exit(pytest.main(["tests/test_pipeline.py", "-v", "--no-header", "-p", "no:cacheprovider"]))
