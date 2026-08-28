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

BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"


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

if FRONTEND_DIR.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIR / "assets"),
        name="assets",
    )

    app.mount(
        "/css",
        StaticFiles(directory=FRONTEND_DIR / "css"),
        name="css",
    )

    app.mount(
        "/js",
        StaticFiles(directory=FRONTEND_DIR / "js"),
        name="js",
    )


# ============================================================
# FRONTEND ROOT
# ============================================================

@app.get("/", include_in_schema=False)
def frontend_root():
    index_file = FRONTEND_DIR / "index.html"

    if index_file.exists():
        return FileResponse(index_file)

    return {
        "message": "Welcome to ResearchGPT API",
        "status": "running",
        "version": "0.1.0",
    }


# ============================================================
# LOGIN PAGE
# ============================================================

@app.get("/login.html", include_in_schema=False)
def frontend_login():
    login_file = FRONTEND_DIR / "login.html"

    if login_file.exists():
        return FileResponse(login_file)

    return {
        "error": "login.html not found"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health", include_in_schema=False)
def health_check():
    return {
        "status": "healthy",
        "service": "ResearchGPT",
    }