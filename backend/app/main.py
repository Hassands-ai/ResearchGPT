from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.api.routes import router

# Database
from app.db.base import Base
from app.db.session import engine

# Import models so SQLAlchemy knows about all tables
from app.models.user import User
from app.models.project import Project
from app.models.paper import Paper


# ============================================================
# PROJECT PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent

FRONTEND_DIR = PROJECT_DIR / "frontend"
FRONTEND_INDEX = FRONTEND_DIR / "index.html"
FRONTEND_LOGIN = FRONTEND_DIR / "login.html"


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def initialize_database():
    """
    Create database tables if they do not already exist.

    This is important for Render because the SQLite database
    may be completely new on the first deployment.
    """
    try:
        Base.metadata.create_all(bind=engine)
        print("DATABASE_INITIALIZED")
    except Exception as exc:
        print(f"DATABASE_INITIALIZATION_WARNING: {exc}")


# ============================================================
# RESEARCHGPT API
# ============================================================

app = FastAPI(
    title="ResearchGPT",
    description="AI-powered research assistant for working with research papers.",
    version="0.1.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():
    initialize_database()


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API ROUTES
# ============================================================

app.include_router(
    router,
    prefix=settings.API_V1_STR,
)


# ============================================================
# FRONTEND STATIC DIRECTORIES
# ============================================================

for folder_name in ["assets", "css", "js"]:
    folder = FRONTEND_DIR / folder_name

    if folder.exists() and folder.is_dir():
        app.mount(
            f"/{folder_name}",
            StaticFiles(directory=str(folder)),
            name=folder_name,
        )


# ============================================================
# FRONTEND ROUTES
# ============================================================

@app.get("/")
def root():
    """
    Main ResearchGPT entry point.
    """
    if FRONTEND_INDEX.exists():
        return FileResponse(str(FRONTEND_INDEX))

    return {
        "message": "Welcome to ResearchGPT API",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/login.html")
def login_page():
    """
    Serve the login page.
    """
    if FRONTEND_LOGIN.exists():
        return FileResponse(str(FRONTEND_LOGIN))

    return {
        "message": "Login page not found",
        "status": "error",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ResearchGPT",
    }


# ============================================================
# API INFORMATION
# ============================================================

@app.get("/api-status")
def api_status():
    return {
        "status": "online",
        "service": "ResearchGPT",
        "api_prefix": settings.API_V1_STR,
    }
