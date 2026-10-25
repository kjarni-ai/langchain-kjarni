"""LangChain's standard integration tests, against a real Kjarni model."""

from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_kjarni import KjarniEmbeddings


class TestKjarniEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> type[KjarniEmbeddings]:
        return KjarniEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "minilm-l6-v2"}
