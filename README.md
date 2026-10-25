# langchain-kjarni

LangChain integration for [Kjarni](https://kjarni.ai): local embeddings in your own process, with no PyTorch, no ONNX Runtime, no GPU and no server.

Kjarni produces the same vectors as sentence-transformers for the same model. The model downloads once on first use and is cached; after that everything works offline.

## Install

```bash
pip install -U langchain-kjarni
```

## Embeddings

```python
from langchain_kjarni import KjarniEmbeddings

embeddings = KjarniEmbeddings(model="minilm-l6-v2")

embeddings.embed_query("What is the capital of Iceland?")      # 384 floats
embeddings.embed_documents(["Reykjavik", "Akureyri"])          # 2 x 384
```

Use it anywhere LangChain takes an `Embeddings`, for example a vector store:

```python
from langchain_core.vectorstores import InMemoryVectorStore

store = InMemoryVectorStore.from_texts(
    ["Reykjavik is the capital of Iceland", "Paris is the capital of France"],
    embedding=embeddings,
)
store.similarity_search("Icelandic capital", k=1)
```

| Setting | Default | |
|---|---|---|
| `model` | `"minilm-l6-v2"` | Any Kjarni embedding model, e.g. `mpnet-base-v2`, `bge-m3` (multilingual). `kjarni model list` shows all. |
| `device` | `"cpu"` | `"gpu"` runs on Vulkan, Metal or DirectX 12. |
| `batch_size` | `32` | Texts per call from `embed_documents`. |
| `query_prefix`, `document_prefix` | `""` | For models trained with prefixes, e.g. `"query: "` and `"passage: "` for E5. |
| `cache_dir` | Kjarni's cache | Where models are stored. |

The model loads on first use, so constructing `KjarniEmbeddings` is free.

## Example

[`examples/rag_without_pytorch.ipynb`](examples/rag_without_pytorch.ipynb) builds a small retrieval pipeline and measures the install size, the vectors against sentence-transformers, and the speed.

## License

MIT
# langchain-kjarni
# langchain-kjarni
