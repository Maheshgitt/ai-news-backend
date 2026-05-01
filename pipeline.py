"""
pipeline.py  —  AI Tech News Pipeline (v3)
─────────────────────────────────────────
Changes vs v2:
  • 10-15 articles output (was 5)
  • Expanded keyword set: OpenAI, Claude, Gemini, AI tools, Google, Meta, etc.
  • Single technical mode — no beginner/engineer split
  • FCM push notification after daily run
  • Returns structured list[dict] instead of printing directly
    so FastAPI can serialize it as JSON
"""

import re
import json
import time
import hashlib
import logging
import os
import requests
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ─── clients ──────────────────────────────────────────────
NEWS_API_KEY   = os.getenv("NEWSDATA_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
FCM_SERVER_KEY = os.getenv("FCM_SERVER_KEY")          # Firebase server key
client         = Groq(api_key=GROQ_API_KEY)

# ─── paths ─────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MEMORY_FILE = BASE_DIR / "memory" / "seen_articles.json"
LOG_DIR     = BASE_DIR / "logs"
MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / f"run_{datetime.now():%Y-%m-%d}.log"),
    ],
)
log = logging.getLogger("pipeline")

# ─── tunables ──────────────────────────────────────────────
MAX_CANDIDATES     = 25    # fetch wide, then filter down
TARGET_ARTICLES    = 12    # LLM picks this many for output
DEDUP_RATIO        = 0.72
MEMORY_EXPIRY_DAYS = 7
GROQ_RETRY         = 3
GROQ_DELAY         = 2

# ──────────────────────────────────────────────────────────
# EXPANDED KEYWORD WEIGHTS
# ──────────────────────────────────────────────────────────
KEYWORD_WEIGHTS: dict[str, int] = {
    # ── AI models & labs (tier S) ─────────────────────────
    "openai": 10, "gpt-5": 10, "gpt-4o": 10, "o3": 10, "o4": 10,
    "chatgpt": 9, "dall-e": 8, "sora": 9, "whisper": 7,
    "claude": 10, "anthropic": 10,
    "gemini": 10, "google deepmind": 10, "bard": 7,
    "grok": 9, "xai": 9, "elon musk ai": 8,
    "llama": 9, "meta ai": 9, "mistral": 9,
    "deepseek": 10, "qwen": 9, "baidu ernie": 8,
    "perplexity": 8, "inflection": 7, "cohere": 7,
    "stability ai": 8, "midjourney": 8, "runway": 8,

    # ── AI concepts ────────────────────────────────────────
    "large language model": 10, "llm": 10,
    "multimodal": 9, "vision language model": 9, "vlm": 9,
    "artificial general intelligence": 10, "agi": 10,
    "reasoning model": 9, "chain of thought": 8,
    "reinforcement learning from human feedback": 9, "rlhf": 9,
    "generative ai": 9, "foundation model": 9,
    "ai agent": 9, "autonomous agent": 9, "agentic ai": 9,
    "retrieval augmented generation": 8, "rag": 8,
    "fine-tuning": 8, "prompt engineering": 7,
    "transformer": 7, "attention mechanism": 8,
    "neural network": 7, "deep learning": 7, "machine learning": 7,
    "computer vision": 7, "natural language processing": 7, "nlp": 7,
    "speech recognition": 6, "text to image": 7, "text to video": 8,
    "diffusion model": 8, "ai hallucination": 7,

    # ── AI tools & products ───────────────────────────────
    "copilot": 8, "github copilot": 9,
    "cursor": 7, "codeium": 7, "tabnine": 6,
    "ai assistant": 7, "ai search": 7,
    "hugging face": 8, "langchain": 7, "llamaindex": 7,

    # ── AI infrastructure / cloud AI ──────────────────────
    "google tpu": 9, "tensor processing unit": 9,
    "nvidia cuda": 8, "nvidia h100": 9, "nvidia b200": 9,
    "aws bedrock": 8, "azure openai": 8, "google vertex ai": 8,

    # ── Semiconductors & hardware ─────────────────────────
    "nvidia": 8, "amd": 7, "intel": 7, "apple silicon": 8,
    "qualcomm": 7, "arm": 7, "tsmc": 8, "samsung foundry": 7,
    "gpu": 7, "cpu": 7, "tpu": 8, "npu": 8, "asic": 7,
    "fpga": 6, "soc": 6, "chip": 5, "processor": 6,
    "semiconductor": 7, "2nm": 9, "3nm": 8,
    "silicon photonics": 8, "hbm memory": 7,
    "quantum chip": 9, "quantum computing": 9,

    # ── Robotics & autonomy ───────────────────────────────
    "humanoid robot": 9, "boston dynamics": 8,
    "figure ai": 9, "1x technologies": 8,
    "self-driving": 8, "autonomous vehicle": 8,
    "tesla autopilot": 8, "waymo": 8, "cruise": 7,
    "robot": 6, "robotics": 6, "automation": 5,
    "drone": 6, "uav": 6,

    # ── Cybersecurity ─────────────────────────────────────
    "zero-day": 10, "cve": 8, "ransomware": 8,
    "cybersecurity": 7, "vulnerability": 7, "exploit": 8,
    "data breach": 8, "cyber attack": 7,

    # ── Networking ────────────────────────────────────────
    "5g": 5, "6g": 7, "starlink": 6,
    "iot": 5, "edge computing": 6,

    # ── Regulation & safety ───────────────────────────────
    "ai regulation": 8, "ai safety": 9, "ai act": 8,
    "ai governance": 7, "copyright ai": 7,

    # ── General ────────────────────────────────────────────
    "breakthrough": 5, "open-source": 5, "benchmark": 5,
    "technology": 1, "startup": 3, "innovation": 2,
}

REJECT_PATTERN = re.compile(
    r"\b(stock price|share price|quarterly earnings|revenue beat|ipo filing"
    r"|market cap|shares surge|dividend|fiscal year|analyst rating"
    r"|merger acquisition|cfo resigns|layoffs count|headcount)\b",
    re.IGNORECASE,
)

# ──────────────────────────────────────────────────────────
# SOURCE CREDIBILITY
# ──────────────────────────────────────────────────────────
SOURCE_SCORES: dict[str, int] = {
    "techcrunch.com": 10, "theverge.com": 10, "wired.com": 10,
    "arstechnica.com": 10, "ieee.org": 10, "nature.com": 10,
    "science.org": 10, "spectrum.ieee.org": 10,
    "venturebeat.com": 8, "zdnet.com": 8, "thenextweb.com": 8,
    "engadget.com": 8, "tomshardware.com": 8,
    "9to5google.com": 7, "macrumors.com": 7,
    "reuters.com": 7, "bloomberg.com": 7,
    "openai.com": 9, "anthropic.com": 9, "deepmind.google": 9,
    "buzzfeed.com": -6, "dailymail.co.uk": -6,
}


# ══════════════════════════════════════════════════════════
# MEMORY
# ══════════════════════════════════════════════════════════
class ArticleMemory:
    def __init__(self, path: Path = MEMORY_FILE):
        self.path  = path
        self._data: dict[str, str] = {}
        self._load()
        self._purge()

    def _load(self):
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except Exception:
                self._data = {}

    def _save(self):
        self.path.write_text(json.dumps(self._data, indent=2))

    def _purge(self):
        cutoff = datetime.now() - timedelta(days=MEMORY_EXPIRY_DAYS)
        self._data = {
            h: ts for h, ts in self._data.items()
            if datetime.fromisoformat(ts) > cutoff
        }
        self._save()

    @staticmethod
    def _hash(title: str) -> str:
        return hashlib.sha1(title.strip().lower().encode()).hexdigest()[:16]

    def seen(self, title: str) -> bool:
        return self._hash(title) in self._data

    def mark_batch(self, titles: list[str]):
        now = datetime.now().isoformat()
        for t in titles:
            self._data[self._hash(t)] = now
        self._save()

    def size(self) -> int:
        return len(self._data)


# ══════════════════════════════════════════════════════════
# FETCH
# ══════════════════════════════════════════════════════════
def fetch_news() -> list[dict]:
    url = "https://newsdata.io/api/1/news"

    params = {
        "apikey": NEWS_API_KEY,
        "category": "technology",
        "language": "en",
        "size": 10
    }

    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()

        data = r.json()
        print("DEBUG:", data)   # keep this for now

        articles = data.get("results", [])
        log.info(f"Fetched {len(articles)} raw articles.")

        return articles

    except Exception as e:
        log.error(f"Fetch failed: {e}")
        return []


# ══════════════════════════════════════════════════════════
# PRE-FILTER
# ══════════════════════════════════════════════════════════
def _kw_score(title: str, desc: str) -> int:
    text = (title + " " + (desc or "")).lower()
    return sum(
        w for kw, w in KEYWORD_WEIGHTS.items()
        if re.search(r"\b" + re.escape(kw) + r"\b", text)
    )

def _src_score(article: dict) -> int:
    url = (article.get("source_url") or article.get("link") or "").lower()
    for domain, bonus in SOURCE_SCORES.items():
        if domain in url:
            return bonus
    return 0

def _is_dup(new_title: str, seen: list[str]) -> bool:
    for s in seen:
        if SequenceMatcher(None, new_title.lower(), s.lower()).ratio() >= DEDUP_RATIO:
            return True
    return False

def prefilter(articles: list[dict], memory: ArticleMemory) -> list[dict]:
    scored: list[tuple[int, dict]] = []
    seen_titles: list[str] = []
    for a in articles:
        title = (a.get("title") or "").strip()
        desc  = (a.get("description") or "").strip()
        if not title or memory.seen(title) or REJECT_PATTERN.search(title):
            continue
        if _is_dup(title, seen_titles):
            continue
        total = _kw_score(title, desc) + _src_score(a)
        if total > 0:
            seen_titles.append(title)
            scored.append((total, a))

    scored.sort(key=lambda x: x[0], reverse=True)
    candidates = [a for _, a in scored[:MAX_CANDIDATES]]
    log.info(f"Pre-filter: {len(candidates)} candidates.")
    return candidates


# ══════════════════════════════════════════════════════════
# LLM CLASSIFIER
# ══════════════════════════════════════════════════════════
CLASSIFIER_SYSTEM = f"""
You are a strict tech news classifier for engineers.

Given a numbered list of articles, select the top {TARGET_ARTICLES} most
impactful ones. Focus on:
  - AI/ML model releases or breakthroughs
  - New hardware (chips, GPUs, quantum)
  - Robotics / autonomy
  - Cybersecurity threats (zero-days, breaches)
  - Regulation with technical impact
  - Open-source model releases

Reject:
  - Pure financial/business news
  - Celebrity/lifestyle
  - Generic company PR

Return ONLY valid JSON, no markdown, no preamble:
{{"selected": [1, 3, 7, ...]}}
""".strip()

def llm_classify(candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    numbered = "\n\n".join(
        f"{i}. {a.get('title','')}\n   {(a.get('description') or '')[:200]}"
        for i, a in enumerate(candidates, 1)
    )
    for attempt in range(1, GROQ_RETRY + 1):
        try:
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": CLASSIFIER_SYSTEM},
                    {"role": "user",   "content": numbered},
                ],
                temperature=0.0,
                max_tokens=150,
            )
            raw  = res.choices[0].message.content.strip()
            data = json.loads(raw)
            indices = [i - 1 for i in data["selected"] if 1 <= i <= len(candidates)]
            selected = [candidates[i] for i in indices[:TARGET_ARTICLES]]
            log.info(f"Classifier selected {len(selected)} articles.")
            return selected
        except Exception as e:
            log.warning(f"Classifier attempt {attempt} failed: {e}")
            if attempt < GROQ_RETRY:
                time.sleep(GROQ_DELAY)
    log.warning("Falling back to keyword top-N.")
    return candidates[:TARGET_ARTICLES]


# ══════════════════════════════════════════════════════════
# SUMMARISER  (technical, no mode split)
# ══════════════════════════════════════════════════════════
SUMMARY_SYSTEM = """
You are an AI tech news summariser writing for software and hardware engineers.

Rules:
- Use correct technical terminology (architecture names, benchmarks, param counts, specs).
- Be precise — include numbers, versions, model sizes, hardware specs where available.
- Zero fluff. No marketing language. No repetition.
- Cover exactly the articles provided, no extras.

Output format (strict):

🔥 Top Trend Today:
[1 sharp sentence capturing the dominant theme]

📰 News ({n} articles):

1. [Article Title]
   Category: [AI Model / Hardware / Cybersecurity / Robotics / Quantum / Regulation / Tools]
   Summary:
   - [What happened — specific, factual]
   - [Technical detail — specs, architecture, benchmark numbers if known]
   - [Who is involved — lab, company, research group]
   - [Context — how this compares to prior state of the art]
   - [Availability — open-source / API / product / paper]
   - [Real-world impact for engineers and builders]
   Why it matters: [1–2 lines. Technical significance, not hype.]

(repeat for all {n} articles)

⚖️ Comparison (only when 2+ articles cover competing systems):
Compare specs, benchmarks, and trade-offs side by side.

📊 Quick Insights:
- Trend Direction: [what's the overall tech direction right now]
- Industry Impact: [who benefits or is disrupted]
- Future Prediction: [next 3–6 months projection, engineering perspective]

❓ Signal or Noise?
Verdict: SIGNAL / NOISE
Reason: [1 sentence — is this week's news moving the field forward or is it hype]
"""

def generate_summary(articles: list[dict]) -> str:
    n = len(articles)
    news_block = "\n\n".join(
        f"{i}. {a.get('title', 'N/A')}\n"
        f"   Description: {a.get('description') or 'N/A'}\n"
        f"   Source: {a.get('source_url') or a.get('link') or 'N/A'}"
        for i, a in enumerate(articles, 1)
    )
    system = SUMMARY_SYSTEM.replace("{n}", str(n))
    for attempt in range(1, GROQ_RETRY + 1):
        try:
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": f"Summarise these {n} tech articles:\n\n{news_block}"},
                ],
                temperature=0.35,
                max_tokens=3000,
            )
            return res.choices[0].message.content
        except Exception as e:
            log.warning(f"Summary attempt {attempt} failed: {e}")
            if attempt < GROQ_RETRY:
                time.sleep(GROQ_DELAY)
    return "[ERROR] Summary generation failed."


# ══════════════════════════════════════════════════════════
# FCM PUSH NOTIFICATION
# ══════════════════════════════════════════════════════════
def send_push_notification(article_count: int):
    """
    Sends an FCM push to the 'tech_news' topic after a successful run.
    All app users subscribed to that topic receive it.
    """
    if not FCM_SERVER_KEY:
        log.warning("FCM_SERVER_KEY not set — skipping push notification.")
        return
    payload = {
        "to": "/topics/tech_news",
        "notification": {
            "title": "Today's Tech News is Ready",
            "body": f"{article_count} new articles — tap to read your daily digest.",
            "sound": "default",
        },
        "data": {
            "type": "daily_digest",
            "article_count": str(article_count),
        },
    }
    try:
        r = requests.post(
            "https://fcm.googleapis.com/fcm/send",
            json=payload,
            headers={
                "Authorization": f"key={FCM_SERVER_KEY}",
                "Content-Type": "application/json",
            },
            timeout=10,
        )
        r.raise_for_status()
        log.info(f"FCM push sent. Response: {r.json()}")
    except Exception as e:
        log.error(f"FCM push failed: {e}")


# ══════════════════════════════════════════════════════════
# MAIN PIPELINE  (called by FastAPI / scheduler)
# ══════════════════════════════════════════════════════════
def run_pipeline() -> dict:
    """
    Returns a dict consumed by FastAPI:
    {
        "run_at": "ISO timestamp",
        "article_count": int,
        "articles": [ {title, description, source_url, link}, ... ],
        "summary": "full markdown summary string",
        "status": "ok" | "error",
        "message": "...",
    }
    """
    memory = ArticleMemory()
    log.info(f"Pipeline started. Memory size: {memory.size()}")

    raw = fetch_news()
    if not raw:
        return {"status": "error", "message": "News API returned no results.", "articles": [], "summary": ""}

    candidates = prefilter(raw, memory)
    if not candidates:
        return {"status": "error", "message": "No new relevant articles (all seen).", "articles": [], "summary": ""}

    selected = llm_classify(candidates)
    if not selected:
        return {"status": "error", "message": "Classifier returned no articles.", "articles": [], "summary": ""}

    summary = generate_summary(selected)

    # Mark seen
    memory.mark_batch([a.get("title", "") for a in selected])

    # Send push
    send_push_notification(len(selected))

    log.info(f"Pipeline complete. {len(selected)} articles.")
    return {
        "status": "ok",
        "run_at": datetime.now().isoformat(),
        "article_count": len(selected),
        "articles": [
            {
                "title":       a.get("title", ""),
                "description": a.get("description", ""),
                "source_url":  a.get("source_url") or a.get("link", ""),
                "image_url":   a.get("image_url", ""),
                "pubDate":     a.get("pubDate", ""),
            }
            for a in selected
        ],
        "summary": summary,
    }