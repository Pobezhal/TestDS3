from sentence_transformers import SentenceTransformer
from typing import List

class BetterEmbeddingFunction:
    """
    Wraps a local sentence-transformers model for Chroma embedding.
    """

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def encode_queries(self, texts: List[str]) -> List[List[float]]:
        # identical to encode_documents for our use case
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.encode_documents(input)