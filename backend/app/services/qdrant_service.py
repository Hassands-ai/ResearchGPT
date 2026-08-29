from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from app.core.config import settings
from typing import List, Dict
import uuid


class QdrantService:
    """
    Lazy-loading Qdrant service.

    Qdrant is NOT contacted when FastAPI starts.
    The connection is created only when an operation actually needs Qdrant.
    """

    def __init__(self):
        self.client = None
        self.collection_name = "papers"

    def _get_client(self):
        """Create the Qdrant client only when it is actually needed."""
        if self.client is None:
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                check_compatibility=False,
            )

        return self.client

    def _ensure_collection(self):
        """Create the papers collection if it does not already exist."""
        client = self._get_client()

        collections = client.get_collections().collections
        exists = any(
            c.name == self.collection_name
            for c in collections
        )

        if not exists:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE,
                ),
            )

    def upsert_chunks(
        self,
        paper_id: int,
        chunks: List[str],
        embeddings: List[List[float]],
    ):
        self._ensure_collection()

        client = self._get_client()

        points = []

        for i, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "paper_id": paper_id,
                        "chunk_index": i,
                        "text": chunk,
                    },
                )
            )

        client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        query_vector: List[float],
        paper_id: int = None,
        limit: int = 5,
    ) -> List[Dict]:

        self._ensure_collection()

        client = self._get_client()

        query_filter = None

        if paper_id is not None:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="paper_id",
                        match=MatchValue(value=paper_id),
                    )
                ]
            )

        response = client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        )

        return [
            {
                "text": point.payload["text"],
                "paper_id": point.payload["paper_id"],
                "score": point.score,
            }
            for point in response.points
        ]


# Lightweight object creation.
# No Qdrant connection is made here.
qdrant_service = QdrantService()
