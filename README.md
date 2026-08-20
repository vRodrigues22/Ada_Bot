# Ada — assistente pessoal no Telegram

Assistente pessoal da Valéria, acessível pelo Telegram no celular, tablet e computador. Roda como um serviço web (FastAPI) hospedado gratuitamente na [Render](https://render.com), usando a [Groq](https://console.groq.com) (gratuita, sem cartão de crédito) como "cérebro" — tanto para conversar em texto quanto para transcrever mensagens de voz.

## Como funciona

1. O Telegram envia cada mensagem para o endpoint `/webhook` deste serviço.
2. Se for texto, a mensagem vai direto para o modelo de chat da Groq (`openai/gpt-oss-120b`).
3. Se for áudio, o arquivo é baixado do Telegram e transcrito pelo Whisper da Groq antes de virar texto.
4. A resposta gerada é enviada de volta ao usuário pelo Telegram.

O histórico da conversa fica em memória (RAM) por `chat_id`, então reinicia se o serviço reiniciar.

## Variáveis de ambiente necessárias

| Nome | Onde conseguir |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Criado com o [@BotFather](https://t.me/BotFather) no Telegram (`/newbot`) |
| `GROQ_API_KEY` | Gratuita em [console.groq.com](https://console.groq.com) → API Keys |

A Render também define `RENDER_EXTERNAL_URL` automaticamente — é usada para registrar o webhook do Telegram no início.

## Deploy

Serviço web Python na Render:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

Depois do primeiro deploy, o próprio serviço configura o webhook do Telegram automaticamente ao subir.

## Próximos passos possíveis

- Persistir o histórico de conversa em um banco (ex.: Render Key Value) em vez de memória.
- Adicionar integração com automação residencial (ex.: Home Assistant).
- Adicionar "ferramentas" (buscas na web, lembretes, agenda).
