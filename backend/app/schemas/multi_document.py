from typing import List

from pydantic import BaseModel, Field


class MultiDocumentSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    paper_ids: List[int] = Field(..., min_length=1)
    limit_per_paper: int = Field(default=5, ge=1, le=20)


class MultiDocumentSearchResult(BaseModel):
    paper_id: int
    text: str
    score: float


class MultiDocumentSearchResponse(BaseModel):
    query: str
    paper_ids: List[int]
    papers_count: int
    results_count: int
    results: List[MultiDocumentSearchResult]