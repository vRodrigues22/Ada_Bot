import os
import logging
from typing import Dict, List

from fastapi import FastAPI, Request
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis-bot")

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]  # gratuito: console.groq.com (sem cartão)

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
TELEGRAM_FILE_API = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

SYSTEM_PROMPT = (
    "Você é a Ada, a assistente pessoal da Valéria, acessível pelo Telegram no "
    "celular, tablet e computador dela. Seja útil, direta, calorosa e um pouco "
    "espirituosa, como uma boa assistente de confiança. Responda sempre em "
    "português do Brasil, de forma natural e objetiva. Se não souber algo, diga "
    "com honestidade em vez de inventar."
)

# Modelo de chat gratuito da Groq (troque aqui se quiser testar outro).
GROQ_CHAT_MODEL = "openai/gpt-oss-120b"
GROQ_WHISPER_MODEL = "whisper-large-v3"
MAX_HISTORY_MESSAGES = 20  # quantidade de mensagens (usuário+assistente) mantidas por conversa

app = FastAPI()

# Memória de conversa em RAM, por chat_id do Telegram.
# Some se o serviço reiniciar — para persistência real, trocar por um banco
# (ex.: Render Key Value) numa fase 2.
conversations: Dict[int, List[dict]] = {}


async def telegram_send_message(chat_id: int, text: str) -> None:
    async with httpx.AsyncClient() as http:
        await http.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text},
        )


async def telegram_send_chat_action(chat_id: int, action: str = "typing") -> None:
    async with httpx.AsyncClient() as http:
        await http.post(
            f"{TELEGRAM_API}/sendChatAction",
            json={"chat_id": chat_id, "action": action},
        )


async def download_telegram_file(file_id: str) -> bytes:
    async with httpx.AsyncClient() as http:
        r = await http.get(f"{TELEGRAM_API}/getFile", params={"file_id": file_id})
        r.raise_for_status()
        file_path = r.json()["result"]["file_path"]
        file_resp = await http.get(f"{TELEGRAM_FILE_API}/{file_path}")
        file_resp.raise_for_status()
        return file_resp.content


async def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcreve áudio usando a API gratuita da Groq (Whisper large-v3)."""
    async with httpx.AsyncClient(timeout=60) as http:
        files = {"file": ("audio.ogg", audio_bytes, "audio/ogg")}
        data = {"model": GROQ_WHISPER_MODEL, "language": "pt"}
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        r = await http.post(GROQ_TRANSCRIBE_URL, headers=headers, data=data, files=files)
        r.raise_for_status()
        return r.json()["text"]


async def ask_groq(chat_id: int, user_text: str) -> str:
    history = conversations.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})
    del history[:-MAX_HISTORY_MESSAGES]  # mantém só as últimas N mensagens

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {"model": GROQ_CHAT_MODEL, "messages": messages, "max_tokens": 1024}

    async with httpx.AsyncClient(timeout=60) as http:
        r = await http.post(GROQ_CHAT_URL, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

    reply = data["choices"][0]["message"]["content"]
    history.append({"role": "assistant", "content": reply})
    return reply or "Desculpa, não consegui gerar uma resposta agora."


@app.get("/")
async def health():
    return {"status": "ok", "bot": "ada"}


@app.post("/webhook")
async def webhook(request: Request):
    update = await request.json()
    message = update.get("message")
    if not message:
        return {"ok": True}

    chat_id = message["chat"]["id"]

    try:
        if "text" in message:
            user_text = message["text"]
        elif "voice" in message:
            await telegram_send_chat_action(chat_id, "typing")
            audio_bytes = await download_telegram_file(message["voice"]["file_id"])
            user_text = await transcribe_audio(audio_bytes)
        else:
            await telegram_send_message(chat_id, "Por enquanto eu só entendo texto e áudio.")
            return {"ok": True}

        await telegram_send_chat_action(chat_id, "typing")
        reply = await ask_groq(chat_id, user_text)
        await telegram_send_message(chat_id, reply)
    except Exception:
        logger.exception("Erro ao processar mensagem do chat %s", chat_id)
        await telegram_send_message(
            chat_id, "Desculpa, tive um problema ao processar isso. Pode tentar de novo?"
        )

    return {"ok": True}


@app.on_event("startup")
async def set_webhook():
    external_url = os.environ.get("RENDER_EXTERNAL_URL")
    if not external_url:
        logger.warning(
            "RENDER_EXTERNAL_URL não definida; configure o webhook manualmente."
        )
        return
    async with httpx.AsyncClient() as http:
        r = await http.post(f"{TELEGRAM_API}/setWebhook", json={"url": f"{external_url}/webhook"})
        logger.info("setWebhook: %s", r.json())
