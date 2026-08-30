from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.api.routes import router


# ============================================================
# PROJECT PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent

FRONTEND_DIR = PROJECT_DIR / "frontend"
FRONTEND_INDEX = FRONTEND_DIR / "index.html"
FRONTEND_LOGIN = FRONTEND_DIR / "login.html"


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
    if FRONTEND_INDEX.exists():
        return FileResponse(str(FRONTEND_INDEX))

    return {
        "message": "Welcome to ResearchGPT API",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/login.html")
def login_page():
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
