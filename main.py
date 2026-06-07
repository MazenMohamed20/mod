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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.0-flash"   # ✅ أحدث وأدق
PORT        = int(os.environ.get("PORT", 8080))
MAX_TOKENS  = 800   # ✅ زيادة عشان الإجابات أكتمل
MAX_Q_LEN   = 600
TIMEOUT_SEC = 45    # ✅ زيادة عشان الأسئلة المعقدة
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
#  FORMAT INSTRUCTIONS — محسّنة
# ============================================================
FORMAT_INSTRUCTIONS = {
    "nutrition": (
        "Reply ONLY with a valid JSON object. No markdown. No explanation. "
        "Use this exact structure:\n"
        '{"food": "name", "calories": 000, "protein": 00, "fat": 00, "carbs": 00}\n'
        "All values must be per 100g unless the user specifies a serving size. "
        "Use USDA FoodData Central as your primary reference. "
        "Protein NEVER exceeds 35g per 100g of any real food. "
        "Fat and carbs must be physiologically realistic. "
        "Double-check all numbers before responding. "
        "Then after the JSON, add: 💡 **You might also ask:** with one related question."
    ),
    "steps": (
        "Reply with clear numbered steps only. Max 8 steps. "
        "Each step must be actionable and specific. No vague instructions. "
        "No intro, no summary. "
        "Then add: 💡 **You might also ask:** or 💡 **قد يهمك أيضاً:** with one related question."
    ),
    "compare": (
        "Reply with a markdown comparison table (max 5 rows, 3 columns). "
        "Use factual, specific data in each cell — no vague terms. "
        "End with **Verdict:** one clear sentence. "
        "Then add: 💡 **You might also ask:** or 💡 **قد يهمك أيضاً:** with one related question."
    ),
    "general": (
        "Reply in 2-5 sentences. Be specific and factual. "
        "Use bullet points if listing 3+ items. "
        "Avoid filler words. Every sentence must add value. "
        "Then add: 💡 **You might also ask:** or 💡 **قد يهمك أيضاً:** with one related question."
    ),
    "yesno": (
        "Start with Yes or No. Then explain in 1-3 sentences with specific reasoning. "
        "If the answer depends on context, say Yes/No but explain the condition. "
        "Then add: 💡 **You might also ask:** or 💡 **قد يهمك أيضاً:** with one related question."
    ),
    "recipe": (
        "Format: **Ingredients** as bullet list with exact quantities, "
        "then **Steps** as numbered list with clear instructions. "
        "Include cooking time and serving size. "
        "Then add: 💡 **You might also ask:** or 💡 **قد يهمك أيضاً:** with one related question."
    ),
    "medical": (
        "Provide accurate medical information based on established medical knowledge. "
        "Be specific and clear. Always end with: "
        "'⚠️ Consult a doctor before making any medical decisions.' "
        "Then add: 💡 **You might also ask:** or 💡 **قد يهمك أيضاً:** with one related question."
    ),
    "fitness": (
        "Provide science-based fitness advice. Include sets, reps, or duration where relevant. "
        "Prioritize safety. Mention any contraindications if relevant. "
        "Then add: 💡 **You might also ask:** or 💡 **قد يهمك أيضاً:** with one related question."
    ),
}

# ============================================================
#  BASE SYSTEM — محسّن للدقة
# ============================================================
BASE_SYSTEM = (
    "You are CuraMind, an advanced AI assistant specialized in health, fitness, nutrition, and general knowledge. "
    "You have expert-level knowledge in: medicine, fitness, nutrition, science, math, history, technology, law, psychology, and cooking. "

    # الدقة
    "ACCURACY RULES: "
    "Always verify numerical values before responding. "
    "Use only scientifically proven information. "
    "Nutritional data must match USDA FoodData Central values. "
    "Medical information must align with WHO and peer-reviewed sources. "
    "If you are not 100% certain of a number, give your best estimate and note it. "

    # جودة الإجابة
    "RESPONSE QUALITY: "
    "Think step by step before answering complex questions. "
    "Be direct, precise, and thorough. "
    "Never use filler phrases like 'Great question' or 'Of course'. "
    "Never repeat the question back to the user. "
    "Every sentence must add value. "
    "Never say 'I don't know' — always give your best reasoned answer. "

    # السلامة
    "SAFETY RULES: "
    "For medical questions, always recommend consulting a doctor at the end. "
    "For fitness questions, prioritize user safety and mention contraindications if relevant. "
    "For nutrition questions, always specify if values are per 100g or per serving. "

    # اللغة
    "LANGUAGE RULES: "
    "Always reply in the SAME language the user writes in. "
    "Arabic question → Arabic answer only. "
    "English question → English answer only. "
    "Never mix languages in the same response. "

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
#  CLASSIFIER — محسّن بكلمات أكتر
# ============================================================
_RULES: list[tuple[str, frozenset]] = [
    ("nutrition", frozenset({
        "calorie","calories","kcal","nutrition","protein","fat","carb","carbs",
        "macro","macros","fiber","sugar","sodium","vitamin","mineral",
        "سعرات","سعر حراري","بروتين","دهون","كربوهيدرات",
        "تغذية","غذائية","قيمة غذائية","كيلو كالوري","ألياف","سكر","فيتامين",
    })),
    ("recipe", frozenset({
        "recipe","cook","bake","prepare","ingredients","ingredient","make","dish",
        "وصفة","اطبخ","حضّر","مكونات","طبخة","اعمل","طبق",
    })),
    ("steps", frozenset({
        "how to","steps","plan","guide","routine","method","tips","procedure",
        "كيف","خطوات","خطة","برنامج","طريقة","نصائح","إجراء",
    })),
    ("compare", frozenset({
        "vs","versus","compare","difference","better","healthier","between","which","difference",
        "مقارنة","الفرق","أفضل","أحسن","بين","أيهما","مقارنه",
    })),
    ("medical", frozenset({
        "symptom","symptoms","disease","diagnosis","treatment","medicine","drug","dose",
        "pain","doctor","hospital","sick","illness","infection","fever","blood",
        "مرض","أعراض","علاج","دواء","جرعة","ألم","طبيب","مستشفى","حمى","دم",
    })),
    ("fitness", frozenset({
        "workout","exercise","gym","muscle","weight","cardio","training","lift",
        "تمرين","رياضة","جيم","عضلة","وزن","كارديو","تدريب",
    })),
    ("yesno", frozenset({
        "is it","can i","should i","do i","does","did","was","were","is there",
        "هل","ممكن","يمكن","هيفيد","يفيد","هيضر","يضر","ينفع","يصلح",
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
#  BUILD PROMPT — محسّن
# ============================================================
def build_prompt(question: str, qtype: str, lang: str) -> str:
    if lang == "arabic":
        lang_instruction = (
            "يجب أن تجيب باللغة العربية الفصحى البسيطة فقط. "
            "لا تستخدم الإنجليزية أبداً. "
            "كن دقيقاً ومحدداً في إجابتك. "
            "تحقق من الأرقام جيداً قبل الإجابة."
        )
    else:
        lang_instruction = (
            "You must reply in English only. Do not use Arabic. "
            "Be specific, accurate, and concise. "
            "Verify all numerical values before responding."
        )

    # استخدم medical أو fitness لو مش موجودين في FORMAT_INSTRUCTIONS
    fmt = FORMAT_INSTRUCTIONS.get(qtype, FORMAT_INSTRUCTIONS["general"])

    return (
        f"{BASE_SYSTEM}\n\n"
        f"LANGUAGE: {lang_instruction}\n\n"
        f"FORMAT: {fmt}\n\n"
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
                            temperature=0.1,   # ✅ منخفض للدقة
                            top_p=0.85,        # ✅ أكثر تركيزاً
                            top_k=20,          # ✅ أقل عشوائية
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


async def on_shutdown(app):
    logger.info("🛑 Server shutting down...")

app = web.Application(middlewares=[cors_middleware])
app.router.add_post("/ask",   handle_question)
app.router.add_get("/health", handle_health)
app.on_shutdown.append(on_shutdown)

if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info(f"  🚀 CuraMind Server | port {PORT}")
    logger.info(f"  🧠 Model   : {GEMINI_MODEL}")
    logger.info(f"  ⏱  Timeout : {TIMEOUT_SEC}s | MaxTok: {MAX_TOKENS}")
    logger.info(f"  🛡  RateLimit: {RATE_LIMIT} req/min per IP")
    logger.info(f"  📍 POST    : http://localhost:{PORT}/ask")
    logger.info(f"  🩺 Health  : http://localhost:{PORT}/health")
    logger.info("=" * 55)
    web.run_app(app, host="0.0.0.0", port=PORT, access_log=None)
