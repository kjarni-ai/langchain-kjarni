"""Behaviour that does not need a real model: a fake stands in for Kjarni."""

import numpy as np
import pytest
from pydantic import ValidationError

from langchain_kjarni import KjarniEmbeddings


class FakeEmbedder:
    """Records what it was asked, and returns a vector per text from its length."""

    def __init__(self, as_numpy: bool) -> None:
        self.as_numpy = as_numpy
        self.batches: list[list[str]] = []
        self.queries: list[str] = []

    def encode(self, text: str) -> list[float]:
        self.queries.append(text)
        return [float(len(text)), 1.0]

    def encode_batch(self, texts):
        self.batches.append(list(texts))
        rows = [[float(len(t)), 1.0] for t in texts]
        return np.array(rows, dtype=np.float32) if self.as_numpy else rows


def with_fake(as_numpy: bool = True, **kwargs) -> tuple[KjarniEmbeddings, FakeEmbedder]:
    embeddings = KjarniEmbeddings(**kwargs)
    fake = FakeEmbedder(as_numpy)
    embeddings._client = fake
    return embeddings, fake


def test_construction_loads_nothing() -> None:
    embeddings = KjarniEmbeddings(model="no-such-model")
    assert embeddings._client is None


@pytest.mark.parametrize("as_numpy", [True, False])
def test_documents_come_back_as_plain_float_lists(as_numpy: bool) -> None:
    embeddings, _ = with_fake(as_numpy)
    vectors = embeddings.embed_documents(["a", "bbb"])
    assert vectors == [[1.0, 1.0], [3.0, 1.0]]
    assert all(type(x) is float for row in vectors for x in row)


def test_batches_and_prefixes() -> None:
    embeddings, fake = with_fake(batch_size=2, query_prefix="query: ", document_prefix="passage: ")
    embeddings.embed_documents(["a", "b", "c"])
    assert fake.batches == [["passage: a", "passage: b"], ["passage: c"]]
    embeddings.embed_query("q")
    assert fake.queries == ["query: q"]


def test_empty_input_is_empty_output() -> None:
    embeddings, fake = with_fake()
    assert embeddings.embed_documents([]) == []
    assert fake.batches == []


def test_bad_settings_are_rejected() -> None:
    with pytest.raises(ValidationError):
        KjarniEmbeddings(batch_size=0)
    with pytest.raises(ValidationError):
        KjarniEmbeddings(device="tpu")
    with pytest.raises(ValidationError):
        KjarniEmbeddings(not_a_setting=1)
