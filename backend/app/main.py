from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.api.routes import router


# ============================================================
# PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
FRONTEND_INDEX = FRONTEND_DIR / "index.html"
FRONTEND_ASSETS = FRONTEND_DIR / "assets"


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
# FRONTEND STATIC FILES
# ============================================================

# Only mount assets if the directory actually exists.
if FRONTEND_ASSETS.exists() and FRONTEND_ASSETS.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_ASSETS)),
        name="assets",
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    # If frontend/index.html exists, serve the frontend.
    if FRONTEND_INDEX.exists():
        return FileResponse(str(FRONTEND_INDEX))

    # Otherwise return API information.
    return {
        "message": "Welcome to ResearchGPT API",
        "status": "running",
        "version": "0.1.0",
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
