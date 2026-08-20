import os
import base64
import logging
from typing import Dict, List

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ada-bot")

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]  # gratuito: console.groq.com (sem cartão)

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
TELEGRAM_FILE_API = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

SYSTEM_PROMPT = (
    "Você é a Ada, a assistente pessoal da Valéria, acessível pelo Telegram e "
    "por um app web instalável no celular, tablet e computador dela. Seja "
    "útil, direta, calorosa e um pouco espirituosa, como uma boa assistente de "
    "confiança. Responda sempre em português do Brasil, de forma natural e "
    "objetiva. Você tem acesso a busca na internet quando precisar de "
    "informações atuais — use esse recurso em vez de inventar respostas. Se "
    "ainda assim não souber algo, diga com honestidade."
)

# Modelo "compound" da Groq: além de conversar, pesquisa na web automaticamente
# quando necessário. Gratuito, sem cartão de crédito.
GROQ_CHAT_MODEL = "openai/gpt-oss-120b"
GROQ_WHISPER_MODEL = "whisper-large-v3"
MAX_HISTORY_MESSAGES = 20  # quantidade de mensagens (usuário+assistente) mantidas por conversa

# Ícones do app (PNG codificado em base64, embutido aqui para não depender de
# upload de arquivos binários — só copiar e colar este main.py já basta).
ICON_192_B64 = "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"
ICON_512_B64 = "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"

ICON_192_BYTES = base64.b64decode(ICON_192_B64)
ICON_512_BYTES = base64.b64decode(ICON_512_B64)

SERVICE_WORKER_JS = """
self.addEventListener('install', (event) => { self.skipWaiting(); });
self.addEventListener('activate', (event) => { event.waitUntil(self.clients.claim()); });
self.addEventListener('fetch', (event) => { event.respondWith(fetch(event.request)); });
"""

CHAT_PAGE_HTML = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Ada</title>
<link rel="manifest" href="/manifest.json">
<link rel="apple-touch-icon" href="/icon-192.png">
<link rel="icon" href="/icon-192.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Ada">
<meta name="theme-color" content="#0f172a">
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    background: #0f172a;
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    display: flex;
    flex-direction: column;
  }
  header {
    padding: max(16px, env(safe-area-inset-top)) 16px 12px;
    display: flex;
    align-items: center;
    gap: 10px;
    border-bottom: 1px solid #1e293b;
    flex-shrink: 0;
  }
  header img { width: 32px; height: 32px; border-radius: 8px; }
  header h1 { font-size: 17px; margin: 0; font-weight: 600; }
  #messages {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .msg { max-width: 82%; padding: 10px 14px; border-radius: 16px; line-height: 1.4; font-size: 15px; white-space: pre-wrap; }
  .msg.user { align-self: flex-end; background: #38bdf8; color: #0f172a; border-bottom-right-radius: 4px; }
  .msg.assistant { align-self: flex-start; background: #1e293b; border-bottom-left-radius: 4px; }
  .msg.typing { align-self: flex-start; background: #1e293b; color: #94a3b8; font-style: italic; }
  form {
    display: flex;
    gap: 8px;
    padding: 12px 16px max(12px, env(safe-area-inset-bottom));
    border-top: 1px solid #1e293b;
    flex-shrink: 0;
  }
  input {
    flex: 1;
    background: #1e293b;
    border: none;
    border-radius: 20px;
    padding: 12px 16px;
    color: #e2e8f0;
    font-size: 15px;
    outline: none;
  }
  button {
    background: #38bdf8;
    color: #0f172a;
    border: none;
    border-radius: 20px;
    padding: 0 20px;
    font-weight: 600;
    font-size: 15px;
    cursor: pointer;
  }
  button:disabled { opacity: 0.5; }
</style>
</head>
<body>
<header>
  <img src="/icon-192.png" alt="Ada">
  <h1>Ada</h1>
</header>
<div id="messages"></div>
<form id="form">
  <input id="input" type="text" placeholder="Fale com a Ada..." autocomplete="off">
  <button type="submit">Enviar</button>
</form>
<script>
  const SESSION_KEY = "ada_session_id";
  let sessionId = localStorage.getItem(SESSION_KEY);
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, sessionId);
  }

  const messagesEl = document.getElementById("messages");
  const formEl = document.getElementById("form");
  const inputEl = document.getElementById("input");

  function addMessage(role, text) {
    const div = document.createElement("div");
    div.className = "msg " + role;
    div.textContent = text;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }

  formEl.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = inputEl.value.trim();
    if (!text) return;
    inputEl.value = "";
    inputEl.disabled = true;
    addMessage("user", text);
    const typingEl = addMessage("typing", "Ada está digitando...");

    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: text }),
      });
      const data = await resp.json();
      typingEl.remove();
      addMessage("assistant", data.reply || "Desculpa, não consegui responder agora.");
    } catch (err) {
      typingEl.remove();
      addMessage("assistant", "Não consegui falar com o servidor. Tenta de novo?");
    } finally {
      inputEl.disabled = false;
      inputEl.focus();
    }
  });

  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("/sw.js").catch(() => {});
  }
</script>
</body>
</html>
"""

app = FastAPI()

# Memória de conversa em RAM, por chave de conversa ("tg:<chat_id>" para
# Telegram, "web:<session_id>" para o app web). Some se o serviço reiniciar —
# para persistência real, trocar por um banco (ex.: Render Key Value) numa
# fase 2.
conversations: Dict[str, List[dict]] = {}


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


async def ask_groq(conversation_key: str, user_text: str) -> str:
    history = conversations.setdefault(conversation_key, [])
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


@app.get("/health")
async def health():
    return {"status": "ok", "bot": "ada"}


@app.get("/", response_class=HTMLResponse)
async def chat_page():
    return CHAT_PAGE_HTML


@app.get("/manifest.json")
async def manifest():
    return JSONResponse({
        "name": "Ada",
        "short_name": "Ada",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#0f172a",
        "theme_color": "#0f172a",
        "icons": [
            {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png"},
        ],
    })


@app.get("/icon-192.png")
async def icon_192():
    return Response(content=ICON_192_BYTES, media_type="image/png")


@app.get("/icon-512.png")
async def icon_512():
    return Response(content=ICON_512_BYTES, media_type="image/png")


@app.get("/sw.js")
async def service_worker():
    return Response(content=SERVICE_WORKER_JS, media_type="application/javascript")


@app.post("/api/chat")
async def api_chat(request: Request):
    body = await request.json()
    session_id = body.get("session_id") or "anon"
    user_text = (body.get("message") or "").strip()
    if not user_text:
        return JSONResponse({"reply": "Manda alguma mensagem pra eu responder :)"})

    try:
        reply = await ask_groq(f"web:{session_id}", user_text)
    except Exception:
        logger.exception("Erro ao processar mensagem web da sessão %s", session_id)
        reply = "Desculpa, tive um problema ao processar isso. Pode tentar de novo?"

    return JSONResponse({"reply": reply})


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
        reply = await ask_groq(f"tg:{chat_id}", user_text)
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
