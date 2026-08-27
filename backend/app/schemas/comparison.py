from typing import List

from pydantic import BaseModel, Field


class PaperComparisonRequest(BaseModel):
    paper_ids: List[int] = Field(
        ...,
        min_length=2,
        max_length=10,
    )

    evidence_per_paper: int = Field(
        default=5,
        ge=1,
        le=10,
    )


class PaperComparisonResponse(BaseModel):
    paper_ids: List[int]
    papers_count: int
    comparison: str
    sources: List[dict]