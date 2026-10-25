"""Writes rag_without_pytorch.ipynb. The install-size table is filled in from
measure_install.sh's output; everything else is computed when it runs."""
import json, sys

sizes = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}

table = "\n".join(f"| {row['what']} | {row['mb']:,} MB | {row['packages']} |" for row in sizes.get("rows", []))
cells = [
md(f"""
# Local RAG in LangChain without PyTorch

`KjarniEmbeddings` runs the same embedding models as `HuggingFaceEmbeddings`, in your own process, without PyTorch. This notebook measures what that changes: the install, whether the vectors really are the same, and the speed. Then it builds a small retriever over Kjarni's own documentation.

Measured on Linux x86_64 (Intel i7-13700, 24 threads), Python 3.13, CPU only. {sizes.get('date', '')}

## 1. What gets installed

A fresh virtual environment for each, measured as the size of `site-packages` after `pip install`:

| Install | Size | Packages |
|:--|--:|--:|
{table}

Without an index URL, `pip` on Linux installs the CUDA build of PyTorch, whether or not the machine has an NVIDIA GPU. [`measure_install.sh`](measure_install.sh) reproduces these numbers.
"""),
md("""
## 2. The same vectors

Both sides run all-MiniLM-L6-v2 on the same texts. If Kjarni is a drop-in replacement, the vectors should agree to floating-point precision.
"""),
code("""
import numpy as np
from langchain_kjarni import KjarniEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

kjarni = KjarniEmbeddings(model="minilm-l6-v2")
torch_st = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

texts = [
    "Reykjavik is the capital of Iceland.",
    "The volcano erupted for three weeks and closed the airport.",
    "Refunds are issued within 14 days of the return.",
    "How do I rerank search results with a cross-encoder?",
    "Kjarni runs embedding models without PyTorch.",
]
a = np.array(kjarni.embed_documents(texts))
b = np.array(torch_st.embed_documents(texts))
cosines = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
print(f"vectors: {a.shape} vs {b.shape}")
print(f"largest difference in any value: {np.abs(a - b).max():.2e}")
print("cosine between the two, per text:", np.round(cosines, 7))
"""),
md("""
## 3. A retriever over Kjarni's documentation

`llms-full.txt` is the whole of kjarni.ai as one Markdown file. Split into passages, indexed in LangChain's `InMemoryVectorStore` with `KjarniEmbeddings`, it answers questions about Kjarni.
"""),
code("""
import urllib.request
from langchain_core.vectorstores import InMemoryVectorStore

request = urllib.request.Request(
    "https://kjarni.ai/llms-full.txt", headers={"User-Agent": "langchain-kjarni-example"}
)
doc = urllib.request.urlopen(request).read().decode("utf-8")

# Paragraphs, merged up to about 400 characters per passage: short
# enough that each passage is about one thing.
passages, current = [], ""
for paragraph in (p.strip() for p in doc.split("\\n\\n")):
    if not paragraph:
        continue
    if current and len(current) + len(paragraph) > 400:
        passages.append(current)
        current = ""
    current = f"{current}\\n\\n{paragraph}" if current else paragraph
passages.append(current)
print(f"{len(doc):,} characters, {len(passages)} passages")

store = InMemoryVectorStore.from_texts(passages, embedding=kjarni)
for question in [
    "Does Kjarni need Python installed?",
    "How do I use Kjarni from a Qt Quick application?",
    "Can it run in the browser?",
]:
    best = store.similarity_search(question, k=1)[0].page_content
    print(f"\\nQ: {question}\\nA: {' '.join(best.split())[:300]}...")
"""),
md("""
## 4. Speed

Embedding every passage from section 3, each side warmed up first. Then cold start: a fresh Python process that imports the integration, loads the model from the local cache and embeds one sentence.
"""),
code("""
import time

for name, model in [("KjarniEmbeddings", kjarni), ("HuggingFaceEmbeddings", torch_st)]:
    model.embed_documents(passages[:8])
    start = time.perf_counter()
    model.embed_documents(passages)
    elapsed = time.perf_counter() - start
    print(f"{name:22} {len(passages)} passages in {elapsed:5.2f} s  ({len(passages) / elapsed:,.0f} per second)")
"""),
code("""
import subprocess, sys

cold = {
    "KjarniEmbeddings": "from langchain_kjarni import KjarniEmbeddings; KjarniEmbeddings().embed_query('hello')",
    "HuggingFaceEmbeddings": (
        "from langchain_huggingface import HuggingFaceEmbeddings; "
        "HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2').embed_query('hello')"
    ),
}
for name, program in cold.items():
    runs = []
    for _ in range(3):
        start = time.perf_counter()
        subprocess.run([sys.executable, "-c", program], check=True, capture_output=True)
        runs.append(time.perf_counter() - start)
    print(f"{name:22} cold start, best of 3: {min(runs):.2f} s")
"""),
md("""
## Summary

`KjarniEmbeddings` is a drop-in replacement for `HuggingFaceEmbeddings`: the same vectors, from an install a fraction of the size, with no PyTorch, no GPU and no server, and a cold start in a fraction of a second rather than seconds. For bulk embedding of a small model like MiniLM on a many-core CPU, PyTorch's throughput is higher, as section 4 shows; where Kjarni pays off is everywhere the process starts often or the install has to stay small: scripts, command-line tools, serverless functions, containers and tests. Other models, including multilingual ones, are listed in the [Kjarni documentation](https://kjarni.ai/docs/).
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("examples/rag_without_pytorch.ipynb", "w"), indent=1, ensure_ascii=False)
print("written")
