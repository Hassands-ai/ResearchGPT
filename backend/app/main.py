from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes import router


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
# ROOT
# ============================================================

@app.get("/")
def root():
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