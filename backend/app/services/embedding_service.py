import requests
from typing import List
from app.core.config import settings


class EmbeddingService:
    """
    Lightweight remote embedding service.

    Embeddings are generated through OpenRouter instead of loading
    SentenceTransformer/PyTorch locally. This keeps the Render
    Free instance below its 512 MB memory limit.
    """

    def __init__(self):
        self.base_url = settings.OPENROUTER_BASE_URL.rstrip("/")
        self.model = "qwen/qwen3-embedding-8b"
        self.dimension = 1024

    def _headers(self):
        api_keys = settings.api_keys_list

        if not api_keys:
            raise RuntimeError(
                "OPENROUTER_API_KEYS is not configured."
            )

        return {
            "Authorization": f"Bearer {api_keys[0]}",
            "Content-Type": "application/json",
        }

    def _embed(self, inputs):
        url = f"{self.base_url}/embeddings"

        response = requests.post(
            url,
            headers=self._headers(),
            json={
                "model": self.model,
                "input": inputs,
            },
            timeout=120,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter embedding request failed "
                f"({response.status_code}): {response.text[:1000]}"
            )

        data = response.json()

        if "data" not in data:
            raise RuntimeError(
                f"Invalid embedding response: {data}"
            )

        embeddings = sorted(
            data["data"],
            key=lambda item: item["index"]
        )

        result = [
            item["embedding"]
            for item in embeddings
        ]

        if not result:
            raise RuntimeError(
                "OpenRouter returned no embeddings."
            )

        return result

    def embed_texts(
        self,
        texts: List[str]
    ) -> List[List[float]]:

        if not texts:
            return []

        return self._embed(texts)

    def embed_query(
        self,
        query: str
    ) -> List[float]:

        result = self._embed([query])

        return result[0]


embedding_service = EmbeddingService()