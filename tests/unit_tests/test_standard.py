"""LangChain's standard unit tests for embeddings."""

from langchain_tests.unit_tests import EmbeddingsUnitTests

from langchain_kjarni import KjarniEmbeddings


class TestKjarniEmbeddingsStandard(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> type[KjarniEmbeddings]:
        return KjarniEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "minilm-l6-v2"}
