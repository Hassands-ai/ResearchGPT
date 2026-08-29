from typing import List
from threading import Lock


class EmbeddingService:
    """
    Lazy-loading embedding service.

    The SentenceTransformer model is NOT loaded when FastAPI starts.
    It is loaded only when embed_texts() or embed_query() is called.
    This keeps Render Free startup memory usage low.
    """

    def __init__(self):
        self.device = "cpu"
        self.model = None
        self._lock = Lock()

    def _load_model(self):
        """Load the embedding model only when it is actually needed."""
        if self.model is None:
            with self._lock:
                if self.model is None:
                    from sentence_transformers import SentenceTransformer

                    self.model = SentenceTransformer(
                        "all-MiniLM-L6-v2",
                        device=self.device,
                    )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        self._load_model()

        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            device=self.device,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        self._load_model()

        embedding = self.model.encode(
            [query],
            show_progress_bar=False,
            device=self.device,
        )[0]

        return embedding.tolist()


# Lightweight object creation.
# The actual AI model is NOT loaded here.
embedding_service = EmbeddingService()
