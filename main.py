# -*- coding: utf-8 -*-
import os
import re
import sys
import time
import asyncio
import logging
from collections import deque
from aiohttp import web
from aiohttp.web_middlewares import middleware

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

logger = logging.getLogger(__name__)

# ============================================================
#  CONFIG
# ============================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")   # ✅ آمن
GEMINI_MODEL   = "gemini-2.5-flash"
PORT        = int(os.environ.get("PORT", 8080))          # ✅ Railway
MAX_TOKENS  = 500
MAX_Q_LEN   = 600
TIMEOUT_SEC = 30
RATE_LIMIT  = 10

_start_time = time.monotonic()
_sem        = asyncio.Semaphore(1)

# ============================================================
#  INIT GEMINI
# ============================================================
gemini_model = None
if not HAS_GENAI:
    logger.error("❌ google-generativeai not installed → pip install google-generativeai")
    sys.exit(1)

if not GEMINI_API_KEY:
    logger.error("❌ GEMINI_API_KEY is not set!")
    sys.exit(1)

try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    logger.info(f"✅ Gemini model loaded | {GEMINI_MODEL}")
except Exception as e:
    logger.error(f"❌ Failed to init Gemini: {e}")
    sys.exit(1)

# ============================================================
#  RATE LIMITER
# ============================================================
_rate_store: dict[str, deque] = {}

def is_rate_limited(ip: str) -> bool:
    now = time.monotonic()
    window = 60.0
    if ip not in _rate_store:
        _rate_store[ip] = deque()
    dq = _rate_store[ip]
    while dq and now - dq[0] > window:
        dq.popleft()
    if len(dq) >= RATE_LIMIT:
        return True
    _rate_store[ip].append(now)
    return False

# ============================================================
#  FORMAT INSTRUCTIONS
# ============================================================
FORMAT_INSTRUCTIONS = {
    "nutrition": (
        "Reply ONLY with a valid JSON object. No markdown. No explanation. "
        "Use this exact structure:\n"
        '{"food": "name", "calories": 000, "protein": 00, "fat": 00, "carbs": 00}\n'
        "All values must be realistic and accurate per 100g or per standard serving. "
        "Protein NEVER exceeds 35g per 100g of any real food. "
        "Then after the JSON, add: 💡 **You might also ask:** with one related question."
    ),
    "steps": (
        "Reply with numbered steps only. Max 8 steps. No intro, no summary. "
        "Then add: 💡 **You might also ask:** or 💡 **قد يهمك أيضاً:** with one related question."
    ),
    "compare": (
        "Reply with a markdown comparison table (max 4 rows, 3 columns). "
        "End with **Verdict:** one sentence. "
        "Then add: 💡 **You might also ask:** or 💡 **قد يهمك أيضاً:** with one related question."
    ),
    "general": (
        "Reply in 2-5 sentences. Use bullet points if listing 3+ items. "
        "Then add: 💡 **You might also ask:** or 💡 **قد يهمك أيضاً:** with one related question."
    ),
    "yesno": (
        "Start with Yes or No. Then explain in 1-3 sentences. "
        "Then add: 💡 **You might also ask:** or 💡 **قد يهمك أيضاً:** with one related question."
    ),
    "recipe": (
        "Format: **Ingredients** as bullet list, then **Steps** as numbered list. "
        "Then add: 💡 **You might also ask:** or 💡 **قد يهمك أيضاً:** with one related question."
    ),
}

BASE_SYSTEM = (
    "You are an advanced AI assistant with expert-level knowledge in all fields: "
    "medicine, fitness, nutrition, science, math, history, technology, law, psychology, cooking, and more. "
    "Always give accurate, detailed, and well-reasoned answers. "
    "Think step by step before answering complex questions. "
    "Always use scientifically accurate nutritional values. "
    "Nutritional data must match real-world values from trusted sources like USDA. "
    "Never refuse to answer. Never say 'I don't know' — always give your best reasoned answer. "
    "Never repeat the question. Never use filler phrases. "
    "Be direct, precise, and thorough. "
    "Always reply in the SAME language the user writes in: "
    "Arabic question → Arabic answer only. English question → English answer only. "
    "Follow the format instruction exactly."
)

# ============================================================
#  LANGUAGE DETECTION
# ============================================================
def detect_language(text: str) -> str:
    arabic_chars  = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    total_letters = sum(1 for c in text if c.isalpha())
    if total_letters == 0:
        return "english"
    return "arabic" if arabic_chars > total_letters * 0.3 else "english"

# ============================================================
#  INPUT SANITIZATION
# ============================================================
def sanitize(text: str) -> str:
    return text.strip()

# ============================================================
#  CLASSIFIER
# ============================================================
_RULES: list[tuple[str, frozenset]] = [
    ("nutrition", frozenset({
        "calorie","calories","kcal","nutrition","protein","fat","carb","carbs",
        "macro","macros","سعرات","سعر حراري","بروتين","دهون","كربوهيدرات",
        "تغذية","غذائية","قيمة غذائية","كيلو كالوري",
    })),
    ("recipe", frozenset({
        "recipe","cook","bake","prepare","ingredients","ingredient",
        "وصفة","اطبخ","حضّر","مكونات","طبخة",
    })),
    ("steps", frozenset({
        "how to","steps","plan","guide","routine","method","tips",
        "كيف","خطوات","خطة","برنامج","طريقة","نصائح",
    })),
    ("compare", frozenset({
        "vs","versus","compare","difference","better","healthier","between","which",
        "مقارنة","الفرق","أفضل","أحسن","بين","أيهما",
    })),
    ("yesno", frozenset({
        "is it","can i","should i","do i","does","did","was","were","is there",
        "هل","ممكن","يمكن","هيفيد","يفيد","هيضر","يضر",
    })),
]

def classify(question: str) -> str:
    q = question.lower()
    tokens = set(q.split())
    for qtype, kws in _RULES:
        for kw in kws:
            if " " in kw:
                if kw in q:
                    return qtype
            elif kw in tokens:
                return qtype
    return "general"

# ============================================================
#  BUILD PROMPT
# ============================================================
def build_prompt(question: str, qtype: str, lang: str) -> str:
    if lang == "arabic":
        lang_instruction = (
            "يجب أن تجيب باللغة العربية فقط. لا تستخدم الإنجليزية أبداً. "
            "فكّر جيداً قبل الإجابة وكن دقيقاً."
        )
    else:
        lang_instruction = (
            "You must reply in English only. Do not use Arabic. "
            "Think carefully and be precise and accurate."
        )

    return (
        f"{BASE_SYSTEM}\n\n"
        f"{lang_instruction}\n\n"
        f"{FORMAT_INSTRUCTIONS[qtype]}\n\n"
        f"Question: {question}"
    )

# ============================================================
#  INFERENCE — Gemini
# ============================================================
async def generate_async(prompt: str, question: str = "") -> str:
    async with _sem:
        try:
            loop = asyncio.get_running_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: gemini_model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=MAX_TOKENS,
                            temperature=0.1,
                            top_p=0.9,
                        )
                    )
                ),
                timeout=TIMEOUT_SEC
            )
            return response.text.strip()
        except asyncio.TimeoutError:
            logger.error(f"⏱ Timeout after {TIMEOUT_SEC}s | question: {question[:50]!r}")
            raise RuntimeError(f"Gemini timed out after {TIMEOUT_SEC}s")
        except Exception as e:
            logger.error(f"💥 Gemini error: {e}")
            raise RuntimeError(str(e))

# ============================================================
#  POST-PROCESS
# ============================================================
def clean_response(text: str) -> str:
    text = text.replace("\\n", "\n").replace("\\t", "\t")
    lines = text.splitlines()
    out, prev_blank = [], False
    for line in lines:
        blank = not line.strip()
        if blank and prev_blank:
            continue
        out.append(line)
        prev_blank = blank
    return "\n".join(out).strip()

# ============================================================
#  CORS MIDDLEWARE
# ============================================================
@middleware
async def cors_middleware(request, handler):
    if request.method == "OPTIONS":
        return web.Response(headers={
            "Access-Control-Allow-Origin":  "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        })
    resp = await handler(request)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

# ============================================================
#  HANDLERS
# ============================================================
async def handle_question(request: web.Request) -> web.Response:
    ip = request.remote or "unknown"
    if is_rate_limited(ip):
        return web.json_response(
            {"error": "Too many requests. Please wait a moment."},
            status=429
        )

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON."}, status=400)

    question = sanitize((body.get("question") or "").strip())

    if not question:
        return web.json_response({"error": "Question is required."}, status=400)
    if len(question) > MAX_Q_LEN:
        return web.json_response(
            {"error": f"Question too long (max {MAX_Q_LEN} chars)."},
            status=400,
        )

    lang   = detect_language(question)
    qtype  = classify(question)
    prompt = build_prompt(question, qtype, lang)

    t0 = time.monotonic()
    logger.info(f"[{qtype.upper()}][{lang}] {question[:80]!r}")

    try:
        raw = await generate_async(prompt, question)
    except RuntimeError as e:
        return web.json_response({"error": str(e)}, status=504)
    except Exception as e:
        logger.exception(f"💥 Inference error | question: {question[:50]!r}")
        return web.json_response({"error": "Gemini inference failed."}, status=500)

    elapsed = time.monotonic() - t0

    if not raw:
        return web.json_response(
            {"error": "Empty response. Try rephrasing."},
            status=500,
        )

    answer = clean_response(raw)
    logger.info(f"[{qtype.upper()}][{lang}] {elapsed:.2f}s → {answer[:80]!r}")

    return web.json_response({
        "response": answer,
        "type":     qtype,
        "lang":     lang,
        "time_ms":  round(elapsed * 1000),
    })


async def handle_health(request: web.Request) -> web.Response:
    health = {
        "status":       "ok",
        "model":        GEMINI_MODEL,
        "timeout":      TIMEOUT_SEC,
        "uptime_sec":   round(time.monotonic() - _start_time),
    }
    if HAS_PSUTIL:
        mem = psutil.Process().memory_info().rss / 1024 / 1024
        health["memory_mb"] = round(mem)
    return web.json_response(health)


# ============================================================
#  GRACEFUL SHUTDOWN
# ============================================================
async def on_shutdown(app):
    logger.info("🛑 Server shutting down...")

# ============================================================
#  APP
# ============================================================
app = web.Application(middlewares=[cors_middleware])
app.router.add_post("/ask",   handle_question)
app.router.add_get("/health", handle_health)
app.on_shutdown.append(on_shutdown)

if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info(f"  🚀 Gemini Server   | port {PORT}")
    logger.info(f"  🧠 Model   : {GEMINI_MODEL}")
    logger.info(f"  ⏱  Timeout : {TIMEOUT_SEC}s | MaxTok: {MAX_TOKENS}")
    logger.info(f"  🛡  RateLimit: {RATE_LIMIT} req/min per IP")
    logger.info(f"  📍 POST    : http://localhost:{PORT}/ask")
    logger.info(f"  🩺 Health  : http://localhost:{PORT}/health")
    logger.info("=" * 55)
    web.run_app(app, host="0.0.0.0", port=PORT, access_log=None)
