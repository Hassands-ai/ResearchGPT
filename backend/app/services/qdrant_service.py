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

    VECTOR_SIZE = 4096

    def __init__(self):

        if settings.QDRANT_URL:
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                check_compatibility=False,
            )
        else:
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                check_compatibility=False,
            )

        self.collection_name = "papers"

        self._ensure_collection()

    def _ensure_collection(self):

        collections = self.client.get_collections().collections

        existing = next(
            (
                collection
                for collection in collections
                if collection.name == self.collection_name
            ),
            None,
        )

        if existing is None:

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )

            return

        # Check existing vector dimension.
        info = self.client.get_collection(
            self.collection_name
        )

        vectors_config = info.config.params.vectors

        existing_size = None

        if hasattr(vectors_config, "size"):
            existing_size = vectors_config.size

        if existing_size != self.VECTOR_SIZE:

            print(
                f"Qdrant collection dimension mismatch: "
                f"{existing_size} != {self.VECTOR_SIZE}"
            )

            print(
                "Recreating collection with "
                f"{self.VECTOR_SIZE}-dimensional vectors."
            )

            self.client.delete_collection(
                collection_name=self.collection_name
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )

    def upsert_chunks(
        self,
        paper_id: int,
        chunks: List[str],
        embeddings: List[List[float]],
    ):

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Chunks and embeddings count mismatch."
            )

        points = []

        for i, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):

            if len(embedding) != self.VECTOR_SIZE:
                raise ValueError(
                    f"Invalid embedding dimension: "
                    f"{len(embedding)}. "
                    f"Expected {self.VECTOR_SIZE}."
                )

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

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    def search(
        self,
        query_vector: List[float],
        paper_id: int = None,
        limit: int = 5,
    ) -> List[Dict]:

        if len(query_vector) != self.VECTOR_SIZE:
            raise ValueError(
                f"Invalid query embedding dimension: "
                f"{len(query_vector)}. "
                f"Expected {self.VECTOR_SIZE}."
            )

        query_filter = None

        if paper_id is not None:

            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="paper_id",
                        match=MatchValue(
                            value=paper_id
                        ),
                    )
                ]
            )

        response = self.client.query_points(
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


qdrant_service = QdrantService()