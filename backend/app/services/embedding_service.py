from sentence_transformers import SentenceTransformer
from typing import List
import torch

class EmbeddingService:
    def __init__(self):
        # Force CPU because GPU is not compatible
        self.device = "cpu"
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False, device=self.device)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query], show_progress_bar=False, device=self.device)[0].tolist()

embedding_service = EmbeddingService()