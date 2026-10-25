"""What makes the integration worth using, checked against the real model."""

import math

from langchain_kjarni import KjarniEmbeddings


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def test_meaning_not_words() -> None:
    embeddings = KjarniEmbeddings()
    query = embeddings.embed_query("doctor")
    near, far = embeddings.embed_documents(["physician", "volcano"])
    assert round(cosine(query, near), 4) == 0.8598
    assert cosine(query, near) > cosine(query, far) + 0.3


def test_query_and_document_paths_agree_without_prefixes() -> None:
    embeddings = KjarniEmbeddings()
    one = embeddings.embed_query("Reykjavik is the capital of Iceland")
    [batch] = embeddings.embed_documents(["Reykjavik is the capital of Iceland"])
    assert cosine(one, batch) > 0.9999


def test_vectors_are_unit_length() -> None:
    [vector] = KjarniEmbeddings().embed_documents(["anything at all"])
    assert abs(math.sqrt(sum(x * x for x in vector)) - 1.0) < 1e-4
