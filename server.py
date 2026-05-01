"""
server.py  —  FastAPI REST server
──────────────────────────────────
Endpoints:
  GET  /api/news      → returns cached latest digest (fast)
  POST /api/refresh   → triggers pipeline immediately (slow, ~30s)
  GET  /health        → liveness check for Render

APScheduler runs the pipeline at DAILY_RUN_TIME automatically.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

from pipeline import run_pipeline

load_dotenv()

log = logging.getLogger("server")

# ─── config ────────────────────────────────────────────────
DAILY_RUN_TIME = os.getenv("DAILY_RUN_TIME", "08:00")   # HH:MM UTC
CACHE_FILE     = Path(__file__).parent / "cache" / "latest.json"
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

# ─── FastAPI app ───────────────────────────────────────────
app = FastAPI(title="AI Tech News API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten this once Android app URL is known
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── cache helpers ─────────────────────────────────────────
def save_cache(data: dict):
    try:
        CACHE_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print("✅ Cache saved successfully:", CACHE_FILE)
    except Exception as e:
        print("❌ Cache save FAILED:", e)

def load_cache() -> dict | None:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

# ─── scheduled job ─────────────────────────────────────────
def scheduled_run():
    log.info("Scheduler triggered pipeline run.")
    result = run_pipeline()
    save_cache(result)
    log.info(f"Scheduled run complete. Status: {result['status']}")

# ─── lifespan (starts scheduler) ──────────────────────────
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    hour, minute = map(int, DAILY_RUN_TIME.split(":"))
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(scheduled_run, "cron", hour=hour, minute=minute)
    scheduler.start()
    log.info(f"Scheduler started — daily run at {DAILY_RUN_TIME} UTC.")
    yield
    scheduler.shutdown()

app = FastAPI(title="AI Tech News API", version="3.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Render uses this to confirm the service is up."""
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/api/news")
def get_news():
    """
    Returns the latest cached digest.
    Android app calls this on startup and on pull-to-refresh.
    Fast response (~10ms) — no API calls.
    """
    cached = load_cache()
    if not cached:
        raise HTTPException(
            status_code=503,
            detail="No digest available yet. POST /api/refresh to trigger the first run.",
        )
    return cached


@app.post("/api/refresh")
async def refresh_news(background_tasks: BackgroundTasks):
    """
    Triggers a fresh pipeline run in the background.
    Returns immediately with a 202; app polls /api/news after a delay.
    """
    background_tasks.add_task(_run_and_cache)
    return {"status": "accepted", "message": "Pipeline started. Check /api/news in ~30 seconds."}


async def _run_and_cache():
    result = run_pipeline()
    save_cache(result)
    log.info(f"Manual refresh complete. Articles: {result.get('article_count', 0)}")