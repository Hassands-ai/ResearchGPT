from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class PaperResponse(BaseModel):
    id: int
    title: str
    file_path: str
    file_size: Optional[int] = None
    status: str
    project_id: Optional[int] = None
    user_id: int
    created_at: datetime
    extracted_text: Optional[str] = None

    class Config:
        from_attributes = True