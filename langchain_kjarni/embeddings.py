"""Kjarni embeddings for LangChain."""

from __future__ import annotations

import threading
from typing import Any, Literal, Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class KjarniEmbeddings(BaseModel, Embeddings):
    """Kjarni embedding models, running locally in your own process.

    Kjarni is a native inference engine with Python bindings. It produces the
    same vectors as sentence-transformers for the same model, without PyTorch,
    ONNX Runtime, a GPU or a server. The model downloads once on first use and
    is cached; after that everything works offline.

    Setup:
        Install ``langchain-kjarni``. There is nothing else to install and no
        API key.

        .. code-block:: bash

            pip install -U langchain-kjarni

    Key init args:
        model: str
            Kjarni model name, for example ``"minilm-l6-v2"`` (384 dimensions,
            the default), ``"mpnet-base-v2"`` (768) or ``"bge-m3"``
            (multilingual, 1024). ``kjarni model list`` shows them all.
        device: Literal["cpu", "gpu"]
            Where to run the model. The GPU path uses Vulkan, Metal or DirectX 12.
        batch_size: int
            Texts per call into the engine from ``embed_documents``.
        query_prefix: str
            Prepended to every query, for models trained with one, such as
            ``"query: "`` for the E5 family.
        document_prefix: str
            Prepended to every document, such as ``"passage: "`` for E5.

    Instantiate:
        .. code-block:: python

            from langchain_kjarni import KjarniEmbeddings

            embeddings = KjarniEmbeddings(model="minilm-l6-v2")

    Embed single text:
        .. code-block:: python

            vector = embeddings.embed_query("What is the capital of Iceland?")
            print(len(vector))  # 384

    Embed multiple texts:
        .. code-block:: python

            vectors = embeddings.embed_documents(["Reykjavik", "Akureyri"])
            print(len(vectors), len(vectors[0]))  # 2 384

    Async:
        .. code-block:: python

            vector = await embeddings.aembed_query("What is the capital of Iceland?")
    """

    model: str = "minilm-l6-v2"
    """Kjarni model name."""
    device: Literal["cpu", "gpu"] = "cpu"
    """Where to run the model."""
    normalize: bool = True
    """L2-normalize vectors, so a dot product is the cosine similarity."""
    batch_size: int = Field(default=32, gt=0)
    """Texts per call into the engine from ``embed_documents``."""
    query_prefix: str = ""
    """Prepended to every query, for models trained with one."""
    document_prefix: str = ""
    """Prepended to every document, for models trained with one."""
    cache_dir: Optional[str] = None
    """Where models are cached. Defaults to Kjarni's own cache directory."""
    quiet: bool = True
    """Suppress Kjarni's progress output, including download progress."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # Loaded on first use rather than at construction, so creating the object
    # is free and never touches the network.
    _client: Any = PrivateAttr(default=None)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def _embedder(self) -> Any:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    from kjarni import Embedder

                    self._client = Embedder(
                        model=self.model,
                        device=self.device,
                        cache_dir=self.cache_dir,
                        normalize=self.normalize,
                        quiet=self.quiet,
                    )
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [self.document_prefix + text for text in texts[start : start + self.batch_size]]
            vectors.extend(_as_lists(self._embedder().encode_batch(batch)))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a query."""
        return [float(x) for x in self._embedder().encode(self.query_prefix + text)]


def _as_lists(vectors: Any) -> list[list[float]]:
    """Kjarni returns a NumPy array when NumPy is installed and lists when it
    is not; LangChain wants plain lists of floats either way."""
    if hasattr(vectors, "tolist"):
        return vectors.tolist()
    return [[float(x) for x in row] for row in vectors]
