from pathlib import Path
from app.services.citation_manager_service import (
    citation_manager_service,
)
from datetime import timedelta
from typing import Optional, List
import uuid

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    UploadFile,
    File,
    Form,
)

from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.session import get_db
from app.models import User, Project, Paper

from app.schemas.user import (
    UserCreate,
    UserResponse,
)

from app.schemas.token import Token

from app.schemas.project import (
    ProjectCreate,
    ProjectResponse,
)

from app.schemas.paper import PaperResponse

from app.schemas.multi_document import (
    MultiDocumentSearchRequest,
    MultiDocumentSearchResponse,
)

from app.schemas.comparison import (
    PaperComparisonRequest,
    PaperComparisonResponse,
)

from app.core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)

from app.api.deps import get_current_user

from app.services.minio_service import minio_service
from app.services.pdf_service import extract_text_from_pdf
from app.services.embedding_service import embedding_service
from app.services.qdrant_service import qdrant_service
from app.services.chunk_service import chunk_text
from app.services.chat_service import chat_service

from app.services.multi_document_service import (
    multi_document_service,
)

from app.services.comparison_service import (
    comparison_service,
)

from app.services.literature_review_service import (
    literature_review_service,
)

from app.services.research_gap_service import (
    research_gap_service,
)

from app.services.paper_writeup_service import (
    paper_writeup_service,
)


router = APIRouter()


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================


class SearchRequest(BaseModel):
    query: str
    paper_id: Optional[int] = None
    limit: int = 5


class SearchResult(BaseModel):
    text: str
    paper_id: int
    score: float


class ChatRequest(BaseModel):
    question: str
    paper_id: int
    limit: int = 5


class LiteratureReviewRequest(BaseModel):
    paper_ids: List[int]
    evidence_per_category: int = 2


class ResearchGapRequest(BaseModel):
    paper_ids: List[int]


class PaperWriteupRequest(BaseModel):
    paper_ids: List[int]
    writeup_type: str = "introduction"
    research_topic: Optional[str] = None
    instructions: Optional[str] = None


# ============================================================
# AUTH
# ============================================================


@router.post(
    "/auth/register",
    response_model=UserResponse,
)
def register(
    user_in: UserCreate,
    db: Session = Depends(get_db),
):
    existing = (
        db.query(User)
        .filter(User.email == user_in.email)
        .first()
    )

    if existing:
        raise HTTPException(
            status_code=400,
            detail="Email already registered",
        )

    user = User(
        email=user_in.email,
        full_name=user_in.full_name,
        hashed_password=get_password_hash(
            user_in.password
        ),
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


@router.post(
    "/auth/login",
    response_model=Token,
)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = (
        db.query(User)
        .filter(
            User.email == form_data.username
        )
        .first()
    )

    if not user or not verify_password(
        form_data.password,
        user.hashed_password,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )

    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        ),
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
    }


@router.get(
    "/auth/me",
    response_model=UserResponse,
)
def get_me(
    current_user: User = Depends(
        get_current_user
    ),
):
    return current_user


# ============================================================
# PROJECTS
# ============================================================


@router.post(
    "/projects",
    response_model=ProjectResponse,
)
def create_project(
    project_in: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):
    project = Project(
        title=project_in.title,
        description=project_in.description,
        user_id=current_user.id,
    )

    db.add(project)
    db.commit()
    db.refresh(project)

    return project


@router.get(
    "/projects",
    response_model=List[ProjectResponse],
)
def list_projects(
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):
    projects = (
        db.query(Project)
        .filter(
            Project.user_id == current_user.id
        )
        .all()
    )

    return projects


# ============================================================
# PAPERS
# ============================================================


@router.get(
    "/papers",
    response_model=List[PaperResponse],
)
def list_papers(
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):
    papers = (
        db.query(Paper)
        .filter(
            Paper.user_id == current_user.id
        )
        .order_by(Paper.id.desc())
        .all()
    )

    return papers


@router.post(
    "/papers/upload",
    response_model=PaperResponse,
)
async def upload_paper(
    file: UploadFile = File(...),
    title: str = Form(...),
    project_id: Optional[int] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed",
        )

    file_data = await file.read()
    file_size = len(file_data)

    if file_size == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty",
        )

    # ------------------------------------------------------------
    # STORE PDF LOCALLY
    # ------------------------------------------------------------
    # Render may not have MinIO configured. Save the uploaded PDF
    # directly to a writable temporary directory instead.
    upload_dir = Path("/tmp/paperaxiom_uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = (
        f"{uuid.uuid4()}_{Path(file.filename).name}"
    )

    local_path = upload_dir / safe_filename

    try:
        local_path.write_bytes(file_data)

        file_path = str(local_path)

    except Exception as e:
        print(f"Local PDF storage failed: {e}")

        raise HTTPException(
            status_code=500,
            detail="Failed to store uploaded PDF.",
        )

    # ------------------------------------------------------------
    # EXTRACT PDF TEXT
    # ------------------------------------------------------------

    try:
        extracted_text = extract_text_from_pdf(
            file_data
        )

        if extracted_text:
            extracted_text = extracted_text.replace(
                "\x00",
                "",
            )

        if not extracted_text or not extracted_text.strip():
            raise ValueError(
                "No readable text could be extracted from PDF."
            )

        status_value = "processed"

    except Exception as e:
        extracted_text = None
        status_value = "uploaded"

        print(
            f"Text extraction failed: {e}"
        )

    paper = Paper(
        title=title,
        file_path=file_path,
        file_size=file_size,
        status=status_value,
        extracted_text=extracted_text,
        project_id=project_id,
        user_id=current_user.id,
    )

    db.add(paper)
    db.commit()
    db.refresh(paper)

    return paper


@router.post(
    "/papers/{paper_id}/process"
)
def process_paper(
    paper_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):
    paper = (
        db.query(Paper)
        .filter(
            Paper.id == paper_id,
            Paper.user_id == current_user.id,
        )
        .first()
    )

    if not paper:
        raise HTTPException(
            status_code=404,
            detail="Paper not found",
        )

    if not paper.extracted_text:
        raise HTTPException(
            status_code=400,
            detail="No extracted text available",
        )

    try:
        chunks = chunk_text(
            paper.extracted_text
        )
    except Exception as e:
        print(
            f"Chunking failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to create text chunks.",
        )

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No text chunks created",
        )

    try:
        embeddings = embedding_service.embed_texts(
            chunks
        )

        qdrant_service.upsert_chunks(
            paper.id,
            chunks,
            embeddings,
        )

    except Exception as e:
        print(
            f"Indexing failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to index paper.",
        )

    paper.status = "indexed"

    db.commit()

    return {
        "message": (
            "Paper processed and indexed successfully"
        ),
        "paper_id": paper.id,
        "chunks_count": len(chunks),
    }


# ============================================================
# PAPER SEARCH
# ============================================================


@router.post(
    "/papers/search",
    response_model=List[SearchResult],
)
def search_papers(
    request: SearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty.",
        )

    if request.limit < 1:
        raise HTTPException(
            status_code=400,
            detail="Limit must be greater than zero.",
        )

    if request.paper_id is not None:
        paper = (
            db.query(Paper)
            .filter(
                Paper.id == request.paper_id,
                Paper.user_id == current_user.id,
            )
            .first()
        )

        if not paper:
            raise HTTPException(
                status_code=404,
                detail="Paper not found",
            )

    query_vector = embedding_service.embed_query(
        request.query
    )

    results = qdrant_service.search(
        query_vector=query_vector,
        paper_id=request.paper_id,
        limit=request.limit,
    )

    return results


# ============================================================
# PAPER CHAT
# ============================================================


@router.post(
    "/papers/chat"
)
def chat_with_paper(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    paper = (
        db.query(Paper)
        .filter(
            Paper.id == request.paper_id,
            Paper.user_id == current_user.id,
        )
        .first()
    )

    if not paper:
        raise HTTPException(
            status_code=404,
            detail="Paper not found",
        )

    try:
        result = chat_service.chat_with_paper(
            question=request.question,
            paper_id=request.paper_id,
            limit=request.limit,
        )

        return result

    except Exception as e:
        print(
            f"Paper chat error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Paper chat failed.",
        )


# ============================================================
# MULTI-DOCUMENT SEARCH
# ============================================================


@router.post(
    "/papers/multi-search",
    response_model=MultiDocumentSearchResponse,
)
def multi_document_search(
    request: MultiDocumentSearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):
    unique_paper_ids = list(
        dict.fromkeys(
            request.paper_ids
        )
    )

    if not unique_paper_ids:
        raise HTTPException(
            status_code=400,
            detail="Select at least one paper.",
        )

    papers = (
        db.query(Paper)
        .filter(
            Paper.id.in_(unique_paper_ids),
            Paper.user_id == current_user.id,
        )
        .all()
    )

    found_paper_ids = {
        paper.id
        for paper in papers
    }

    unauthorized_ids = [
        paper_id
        for paper_id in unique_paper_ids
        if paper_id not in found_paper_ids
    ]

    if unauthorized_ids:
        raise HTTPException(
            status_code=404,
            detail="One or more selected papers could not be found.", 
        )

    try:
        return multi_document_service.search(
            query=request.query,
            paper_ids=unique_paper_ids,
            limit_per_paper=(
                request.limit_per_paper
            ),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        print(
            f"Multi-document search error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Multi-document search failed",
        )


# ============================================================
# COUNTS
# ============================================================


@router.get(
    "/users/count"
)
def get_users_count(
    db: Session = Depends(get_db),
):
    return {
        "users_count": db.query(User).count()
    }


@router.get(
    "/projects/count"
)
def get_projects_count(
    db: Session = Depends(get_db),
):
    return {
        "projects_count": db.query(Project).count()
    }


@router.get(
    "/papers/count"
)
def get_papers_count(
    db: Session = Depends(get_db),
):
    return {
        "papers_count": db.query(Paper).count()
    }


# ============================================================
# DELETE PAPER
# ============================================================


@router.delete(
    "/papers/{paper_id}"
)
def delete_paper(
    paper_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):
    paper = (
        db.query(Paper)
        .filter(
            Paper.id == paper_id,
            Paper.user_id == current_user.id,
        )
        .first()
    )

    if not paper:
        raise HTTPException(
            status_code=404,
            detail="Paper not found",
        )

    db.delete(paper)
    db.commit()

    return {
        "message": "Paper deleted successfully",
        "id": paper_id,
    }


# ============================================================
# PAPER COMPARISON
# ============================================================


@router.post(
    "/papers/compare",
    response_model=PaperComparisonResponse,
)
def compare_papers(
    request: PaperComparisonRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):
    unique_paper_ids = list(
        dict.fromkeys(
            request.paper_ids
        )
    )

    if len(unique_paper_ids) < 1:
        raise HTTPException(
            status_code=400,
            detail="Select at least one paper for analysis.",
        )

    if len(unique_paper_ids) > 10:
        raise HTTPException(
            status_code=400,
            detail="You can select a maximum of 10 papers.",
        )

    if len(unique_paper_ids) < 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "Select at least one paper for comparison."
            ),
        )

    papers = (
        db.query(Paper)
        .filter(
            Paper.id.in_(unique_paper_ids),
            Paper.user_id == current_user.id,
        )
        .all()
    )

    found_paper_ids = {
        paper.id
        for paper in papers
    }

    missing_ids = [
        paper_id
        for paper_id in unique_paper_ids
        if paper_id not in found_paper_ids
    ]

    if missing_ids:
        raise HTTPException(
            status_code=404,
            detail="One or more selected papers could not be found.", 
        )

    try:
        return comparison_service.compare(
            paper_ids=unique_paper_ids,
            evidence_per_paper=(
                request.evidence_per_paper
            ),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        print(
            f"Paper comparison error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Paper comparison failed.",
        )


# ============================================================
# LITERATURE REVIEW
# ============================================================


@router.post(
    "/papers/literature-review"
)
def generate_literature_review(
    request: LiteratureReviewRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):
    """
    Generate an evidence-grounded academic
    literature review from selected papers.
    """

    unique_paper_ids = list(
        dict.fromkeys(
            request.paper_ids
        )
    )

    if len(unique_paper_ids) < 1:
        raise HTTPException(
            status_code=400,
            detail="Select at least one paper for analysis.",
        )

    if len(unique_paper_ids) > 10:
        raise HTTPException(
            status_code=400,
            detail="You can select a maximum of 10 papers.",
        )

    if len(unique_paper_ids) < 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "Select at least one paper for literature review."
            ),
        )

    papers = (
        db.query(Paper)
        .filter(
            Paper.id.in_(unique_paper_ids),
            Paper.user_id == current_user.id,
        )
        .all()
    )

    found_paper_ids = {
        paper.id
        for paper in papers
    }

    missing_ids = [
        paper_id
        for paper_id in unique_paper_ids
        if paper_id not in found_paper_ids
    ]

    if missing_ids:
        raise HTTPException(
            status_code=404,
            detail="One or more selected papers could not be found.", 
        )

    try:
        # IMPORTANT:
        # LiteratureReviewService.generate()
        # currently accepts paper_ids only.
        #
        # Do NOT pass evidence_per_category.
        result = literature_review_service.generate(
            paper_ids=unique_paper_ids
        )

        return result

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        print(
            f"Literature review error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Literature review generation failed."
            ),
        )


# ============================================================
# RESEARCH GAP ANALYSIS
# ============================================================


@router.post(
    "/papers/research-gap"
)
def generate_research_gap(
    request: ResearchGapRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):
    """
    Generate an evidence-grounded research-gap
    analysis from multiple selected research papers.
    """

    unique_paper_ids = list(
        dict.fromkeys(
            request.paper_ids
        )
    )

    if len(unique_paper_ids) < 1:
        raise HTTPException(
            status_code=400,
            detail="Select at least one paper for analysis.",
        )

    if len(unique_paper_ids) > 10:
        raise HTTPException(
            status_code=400,
            detail="You can select a maximum of 10 papers.",
        )

    if len(unique_paper_ids) < 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "Select at least one paper for research-gap analysis."
            ),
        )

    papers = (
        db.query(Paper)
        .filter(
            Paper.id.in_(unique_paper_ids),
            Paper.user_id == current_user.id,
        )
        .all()
    )

    found_paper_ids = {
        paper.id
        for paper in papers
    }

    missing_ids = [
        paper_id
        for paper_id in unique_paper_ids
        if paper_id not in found_paper_ids
    ]

    if missing_ids:
        raise HTTPException(
            status_code=404,
            detail="One or more selected papers could not be found.", 
        )

    try:
        result = research_gap_service.generate(
            paper_ids=unique_paper_ids
        )

        return result

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        print(
            f"Research gap error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Research gap generation failed."
            ),
        )
# ============================================================
# CITATION MANAGER
# ============================================================

@router.post(
    "/papers/citation-manager"
)
def generate_citation_manager(
    request: ResearchGapRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):
    """
    Generate evidence-grounded citation information
    for 1–10 selected research papers.
    """

    unique_paper_ids = list(
        dict.fromkeys(
            request.paper_ids
        )
    )

    if not unique_paper_ids:
        raise HTTPException(
            status_code=400,
            detail="Select at least one paper.",
        )

    if len(unique_paper_ids) > 10:
        raise HTTPException(
            status_code=400,
            detail=(
                "You can select a maximum "
                "of 10 papers."
            ),
        )

    papers = (
        db.query(Paper)
        .filter(
            Paper.id.in_(unique_paper_ids),
            Paper.user_id == current_user.id,
        )
        .all()
    )

    found_ids = {
        paper.id
        for paper in papers
    }

    missing_ids = [
        paper_id
        for paper_id in unique_paper_ids
        if paper_id not in found_ids
    ]

    if missing_ids:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Paper(s) not found: "
                f"{missing_ids}"
            ),
        )

    try:

        return citation_manager_service.generate(
            paper_ids=unique_paper_ids
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        print(
            "Citation Manager error:",
            exc,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Citation Manager generation failed."
            ),
        )

# ============================================================
# PAPER WRITE-UP
# ============================================================


@router.post(
    "/papers/writeup"
)
def generate_paper_writeup(
    request: PaperWriteupRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):
    """Generate an evidence-grounded academic paper section."""

    unique_paper_ids = list(
        dict.fromkeys(
            int(paper_id)
            for paper_id in request.paper_ids
        )
    )

    if not unique_paper_ids:
        raise HTTPException(
            status_code=400,
            detail="Select at least one paper.",
        )

    if len(unique_paper_ids) > 10:
        raise HTTPException(
            status_code=400,
            detail="You can select a maximum of 10 papers.",
        )

    papers = (
        db.query(Paper)
        .filter(
            Paper.id.in_(unique_paper_ids),
            Paper.user_id == current_user.id,
        )
        .all()
    )

    found_ids = {paper.id for paper in papers}
    missing_ids = [
        paper_id
        for paper_id in unique_paper_ids
        if paper_id not in found_ids
    ]

    if missing_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Paper(s) not found: {missing_ids}",
        )

    try:
        return paper_writeup_service.generate(
            paper_ids=unique_paper_ids,
            writeup_type=request.writeup_type,
            research_topic=request.research_topic,
            instructions=request.instructions,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        print(
            "Paper Write-up error:",
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail="Paper write-up generation failed.",
        )

