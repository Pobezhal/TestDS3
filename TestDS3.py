# not a malicious thing, just a bot to make fun of my close friends!!
import asyncio  # Добавить в начале файла

from pathlib import Path
import time
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import Update
from pdfminer.high_level import extract_text  # Для PDF
from docx import Document  # Для DOCX
import requests
import os
from PERSONAS import Persona, PERSONAS
from telegram import Update
import random
from telegram.ext import filters
from dotenv import load_dotenv
import logging
from collections import defaultdict, deque
from enum import Enum, auto
import base64
from io import BytesIO
from openai import OpenAI
from textblob import TextBlob
from persona_hooks import PERSONA_HOOKS
import sys

#chat_memories = defaultdict(lambda: deque(maxlen=32))
persona_contexts = defaultdict(dict)





# def switch_persona(chat_id: int, user_id: int, new_persona: Persona) -> dict:
#     key = (chat_id, user_id, new_persona.value)
#     if key not in persona_contexts:
#         persona_contexts[key] = {
#             "message_history": deque(maxlen=32),
#             "hooks": {},
#             "sentiment": 0.0
#         }
#     return persona_contexts[key]  # Всегда возвращает контекст

def switch_persona(chat_id: int, user_id: int, new_persona: Persona) -> dict:
    key = (chat_id, user_id, new_persona.value)
    if key not in persona_contexts:
        persona_contexts[key] = {
            "message_history": deque(maxlen=32),
            "user_hooks": {},  # Changed from "hooks" to "user_hooks"
            "bot_hooks": {},   # Added to match handle_mention
            "sentiment": 0.0
        }
    return persona_contexts[key]


# Load tokens
load_dotenv()






print("OPENAI_KEY_EXISTS:", "OPENAI_API_KEY" in os.environ)  # Debug line
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



chat_modes = defaultdict(lambda: "normal")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize bot
app = Application.builder().token(os.getenv("BOT_TOKEN")).build()

# DeepSeek API setup
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_HEADERS = {
    "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
    "Content-Type": "application/json"
}


# --------------------------------------
# EXACT FUNCTIONS FROM YOUR LIST (NO MORE, NO LESS)
# --------------------------------------

def decay_hooks(hooks: dict) -> dict:
    return {k: (v - 0.5 if v >= 3 else v) for k, v in hooks.items()}


async def set_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat.id
    user_id = update.effective_user.id  # <- ДОБАВЛЕНО для работы с persona_contexts

    if not context.args:
        await update.message.reply_text("Доступные режимы:\n" + "\n".join([f"- {p.value}" for p in Persona]))
        return

    mode_name = context.args[0].lower()
    try:
        persona = Persona(mode_name)
        # ДОБАВЛЕНО: Создаем/загружаем контекст новой персоны
        switch_persona(chat_id, user_id, persona)  # <- НОВАЯ СТРОКА
        chat_modes[chat_id] = persona.value
        await update.message.reply_text(f"🔹 Режим '{persona.value}' включён")
    except ValueError:
        await update.message.reply_text("❌ Неизвестный режим. Доступные: " + ", ".join([p.value for p in Persona]))

async def news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat.id
    persona_config = PERSONAS[Persona(chat_modes[chat_id])]

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": persona_config["system"]
            },
            {
                "role": "user",
                "content": "Как Владимир Жириновский, энергично (4-5 предложений) изложи ОДНУ свежую политическую новость из Америки или Европы, встроив саркастичный/едкий комментарий прямо в текст. Формат: [Факт новости], [циничный анализ]. [Ещё один факт], [язвительное замечание]"
            }
        ],
        "temperature": persona_config["temperature"]
    }

    response = await call_deepseek(payload)
    await update.message.reply_text(response[:700])


async def wtf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat.id
    persona_config = PERSONAS[Persona(chat_modes[chat_id])]

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": persona_config["system"]},
            {"role": "user", "content": "Объясни смысл жизни (макс. 4 предложения)"}
        ],
        "temperature": persona_config["temperature"],
        "max_tokens": 500
    }
    await update.message.reply_text(await call_deepseek(payload))


async def problem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Не вижу проблемы")
        return

    chat_id = update.message.chat.id
    user_problem = " ".join(context.args)

    # Получаем конфиг персонажа
    persona_config = PERSONAS[Persona(chat_modes[chat_id])]

    # Генерируем payload
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": persona_config["system"]
            },
            {
                "role": "user",
                "content": f"Дай совет по проблеме: {user_problem}"
            }
        ],
        "temperature": persona_config["temperature"]
    }

    # Отправляем
    response = await call_deepseek(payload)
    await update.message.reply_text(response)


async def fugoff(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Послать нахуй"""
    chat_id = update.message.chat.id
    target = context.args[0] if context.args and context.args[0].startswith("@") else "Всем петушкам в чатике"
    prompt = f"Придумай ОДНО креативное оскорбление для {target} (макс. 3 предложения). Начинай с маленькой буквы. Используй мат и сарказм и зумерский лексикон. Не используй кавычки!"
    payload = build_prompt(chat_id, prompt, chat_modes[chat_id])
    await update.message.reply_text(f"{target}, {await call_deepseek(payload)} 🖕")


async def randomeme(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Случайный мем"""
    chat_id = update.message.chat.id
    prompt = "Сгенерируй ОДИН случайный мем/шутку с цинизмом и черным юмором (макс. 3 предложения)"
    payload = build_prompt(chat_id, prompt, chat_modes[chat_id])
    await update.message.reply_text(await call_deepseek(payload))


async def sych(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Оправдание одиночества"""
    chat_id = update.message.chat.id
    prompt = "Объясни почему тян не нужны, а быть одиноким сычем - классно (3 предложения, цинично)"
    payload = build_prompt(chat_id, prompt, chat_modes[chat_id])
    await update.message.reply_text(await call_deepseek(payload))


async def petros(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Шутка Петросяна"""
    chat_id = update.message.chat.id
    prompt = "Придумай одну единственную шутку в стиле Евгения Петросяна (макс. 3 предложения, глупо и смешно)"
    payload = build_prompt(chat_id, prompt, chat_modes[chat_id])
    await update.message.reply_text(await call_deepseek(payload))


async def putin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Цитата Путина"""
    chat_id = update.message.chat.id
    prompt = "Придумай ОДНУ единственную фразу в стиле Владимира Путина (макс. 2 предложения, смело и патриотично)"
    payload = build_prompt(chat_id, prompt, chat_modes[chat_id])
    await update.message.reply_text(await call_deepseek(payload))


async def zhir(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Цитата Жириновского"""
    chat_id = update.message.chat.id
    prompt = "Придумай ОДНУ единственную резкую фразу в стиле Владимира Жириновского (макс. 2 предложения)"
    payload = build_prompt(chat_id, prompt, chat_modes[chat_id])
    await update.message.reply_text(await call_deepseek(payload))


async def hohly(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Новости про Украину"""
    chat_id = update.message.chat.id
    prompt = "Кратко и цинично объясни 'че там у хохлов' (3 предложения)"
    payload = build_prompt(chat_id, prompt, chat_modes[chat_id])
    await update.message.reply_text(await call_deepseek(payload))


async def sage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Восточная мудрость"""
    chat_id = update.message.chat.id
    prompt = "Придумай ОДНУ единственную очень мудрую и глубокую по смыслу фразу в стиле восточной мудрости (макс. 3 предложения)"
    payload = build_prompt(chat_id, prompt, chat_modes[chat_id])
    await update.message.reply_text(await call_deepseek(payload))


async def watts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Цитата Уоттса"""
    chat_id = update.message.chat.id
    prompt = "Придумай ОДНУ единственную очень глубокую и мудрую фразу в стиле философа Алана Уоттса (макс. 3 предложения)"
    payload = build_prompt(chat_id, prompt, chat_modes[chat_id])
    await update.message.reply_text(await call_deepseek(payload))


# --------------------------------------
# UTILITY FUNCTION
# --------------------------------------
def build_prompt(
        chat_id: int,
        user_input: str,
        persona_name: str,
        user_id: int = None
) -> dict:
    persona = PERSONAS.get(Persona(persona_name), PERSONAS[Persona.NORMAL])
    context = persona_contexts.get(
        (chat_id, user_id, persona_name),
        {"message_history": deque(maxlen=32), "user_hooks": {}, "sentiment": 0.0}
    )

    # Format history with sender/persona tags
    history_str = "\n".join(
        f"{msg['sender']} ({msg.get('persona', 'user')}): {msg['text']}"
        for msg in context["message_history"]
    )

    # Get active hooks (user-only, min 2 triggers)
    active_hooks = [f"{k}({v})" for k, v in context["user_hooks"].items() if v >= 2]

    # NEW: Dynamic mood detection
    mood = "🔥 АГРЕССИВНЫЙ" if context.get("sentiment", 0) < -0.3 else \
        "😊 ДОВОЛЬНЫЙ" if context.get("sentiment", 0) > 0.3 else \
            "😐 НЕЙТРАЛЬНЫЙ"

    return {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": (
                    f"{persona['system']}\n\n"
                    f"СОСТОЯНИЕ ПОЛЬЗОВАТЕЛЯ: {mood}\n"  # Injected here
                    f"ТРИГГЕРЫ: {', '.join(active_hooks) if active_hooks else 'нет'}\n"
                    f"ПОСЛЕДНИЕ СООБЩЕНИЯ:\n{history_str[-800:]}"
                )
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        "temperature": persona["temperature"],
        "max_tokens": 700,
        "frequency_penalty": 1
    }


async def call_deepseek(payload: dict) -> str:
    """Call DeepSeek API with nuclear-grade quote prevention"""
    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers=DEEPSEEK_HEADERS,
            json=payload,
            timeout=20  # Увеличенный таймаут
        )
        response.raise_for_status()

        raw_text = response.json()['choices'][0]['message']['content'].strip()

        # Strip edge quotes only
        if raw_text.startswith(('"', "'", "«")):
            raw_text = raw_text[1:]
        if raw_text.endswith(('"', "'", "»")):
            raw_text = raw_text[:-1]

        return raw_text or "Что-то не вышло. Давай еще."  # Сохраняем старый фолбек

    except requests.exceptions.RequestException as e:
        logger.error(f"API Error: {e}")
        if isinstance(e, requests.exceptions.Timeout):
            return "Сервер барахлит. Подожди минуту."  # Старая формулировка
        if response.status_code == 429:  # type: ignore
            return "Слишком много запросов. Остынь."
        return "API сдох. Попробуй позже."

    except Exception as e:
        logger.critical(f"Critical: {e}")
        return "Поддержка уже работает над проблемой... Позвоните в OpenAI."  # Старый фолбек


    # --------------------------------------


# GROUP MENTION HANDLER (SPECIAL CASE)
# --------------------------------------


async def handle_mention(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    chat_id = update.message.chat.id
    user_id = update.effective_user.id
    current_persona = Persona(chat_modes.get(chat_id, "normal"))
    text = update.message.text

    # 1. Load or create persona context
    persona_ctx = persona_contexts.setdefault(
        (chat_id, user_id, current_persona.value),
        {
            "message_history": deque(maxlen=32),
            "user_hooks": {},
            "sentiment": 0.0
        }
    )

    # 2. Update hooks from user message (raw counts)
    for trigger in PERSONA_HOOKS.get(current_persona, []):
        if any(trigger in word.lower() for word in text.split()):
            persona_ctx["user_hooks"][trigger] = persona_ctx["user_hooks"].get(trigger, 0) + 1

    # 3. Decay ONLY hooks that reached threshold (>=3)
    persona_ctx["user_hooks"] = {
        k: (v - 0.5 if v >= 3 else v)  # Halve if mature, else preserve
        for k, v in persona_ctx["user_hooks"].items()
    }

    # 4. Store message with metadata
    persona_ctx["message_history"].append({
        "text": text,
        "sender": "user",
        "persona": None
    })

    # 5. Prepare active hooks (ONLY those >=2 to influence response)
    active_hooks = [k for k, v in persona_ctx["user_hooks"].items() if v >= 2]
    hook_context = f"[User Triggers: {', '.join(active_hooks)}]\n" if active_hooks else ""

    # 6. Generate response
    payload = build_prompt(
        chat_id=chat_id,
        user_input=f"{hook_context}{text}",
        persona_name=current_persona.value,
        user_id=user_id
    )
    response = await call_deepseek(payload)

    # 7. Store bot response
    persona_ctx["message_history"].append({
        "text": response,
        "sender": "bot",
        "persona": current_persona.value
    })

    await update.message.reply_text(response)

async def handle_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.reply_to_message.from_user.id == context.bot.id:
        return

    chat_id = update.message.chat.id
    user_id = update.effective_user.id
    current_persona = Persona(chat_modes.get(chat_id, "normal"))

    # Use structured history instead of chat_memories
    persona_ctx = switch_persona(chat_id, user_id, current_persona)
    persona_ctx["message_history"].append({
        "text": update.message.text,
        "sender": "user",
        "persona": None
    })

    payload = build_prompt(
        chat_id=chat_id,
        user_input=update.message.text,
        persona_name=current_persona.value,
        user_id=user_id
    )
    response = await call_deepseek(payload)
    await update.message.reply_text(response)

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.photo or (update.message.document and update.message.document.mime_type.startswith('image/')):
        await handle_image(update, context)
        return

    # Validate file type (existing)
    ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt", ".csv"]
    try:
        file_ext = os.path.splitext(update.message.document.file_name)[1].lower()
    except AttributeError:
        await update.message.reply_text("❌ Не могу определить тип файла.")
        return

    if file_ext not in ALLOWED_EXTENSIONS:
        await update.message.reply_text("❌ Только PDF/DOCX/TXT/CSV.")
        return

    # Download (existing)
    progress_msg = await update.message.reply_text("📥 Загружаю файл...")
    file_path = f"/tmp/{int(time.time())}_{update.message.document.file_name}"

    try:
        for attempt in range(3):
            try:
                telegram_file = await update.message.document.get_file()
                await telegram_file.download_to_drive(custom_path=file_path)
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    break
                elif attempt == 2:
                    await progress_msg.edit_text("💥 Файл не скачался.")
                    return
            except Exception as e:
                if attempt == 2:
                    await progress_msg.edit_text("💥 Ошибка загрузки.")
                    return
                await asyncio.sleep(1)

        # NEW: Load persona context
        chat_id = update.message.chat.id
        user_id = update.effective_user.id
        current_persona = Persona(chat_modes.get(chat_id, "normal"))
        persona_ctx = switch_persona(chat_id, user_id, current_persona)

        # NEW: Store file metadata
        persona_ctx["message_history"].append({
            "text": f"[File: {update.message.document.file_name}]",
            "sender": "user",
            "persona": None
        })

        # Parse content (existing)
        text = ""
        if file_ext == ".pdf":
            try:
                text = extract_text(file_path)
            except Exception as e:
                logger.error(f"PDF Error: {e}")
                await progress_msg.edit_text("🤖 Не смог прочитать PDF")
                return
        elif file_ext == ".docx":
            doc = Document(file_path)
            text = "\n".join(p.text for p in doc.paragraphs if p.text)
        elif file_ext in (".txt", ".csv"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read(3000)

        if not text.strip():
            await progress_msg.edit_text("🤷‍♂️ Файл пустой или нечитаемый.")
            return

        # Generate summary (existing)
        payload = build_prompt(
            chat_id=chat_id,
            user_input=f"Резюме документа (3 предложения):\n{text}",
            persona_name=current_persona.value,
            user_id=user_id  # NEW: Added user_id
        )
        summary = await call_deepseek(payload)

        # NEW: Store bot response
        persona_ctx["message_history"].append({
            "text": summary,
            "sender": "bot",
            "persona": current_persona.value
        })

        await progress_msg.edit_text(f"📄 Вывод:\n{summary[:1000]}")

    except Exception as e:
        logger.error(f"FILE ERROR: {str(e)}", exc_info=True)
        await progress_msg.edit_text("💥 Ошибка обработки. Попробуй другой файл.")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.photo:
        await update.message.reply_text("Это не изображение.")
        return

    try:
        # --- Original Image Processing (UNTOUCHED) ---
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = BytesIO()
        await photo_file.download_to_memory(out=photo_bytes)
        base64_image = base64.b64encode(photo_bytes.getvalue()).decode('utf-8')

        user_question = (
            update.message.caption or
            (update.message.reply_to_message.text if update.message.reply_to_message else None) or
            "Опиши это изображение. "
        )

        # --- NEW: Structured Memory Integration ---
        chat_id = update.message.chat.id
        user_id = update.effective_user.id
        current_persona = Persona(chat_modes.get(chat_id, "normal"))  # Your original persona logic
        persona_ctx = switch_persona(chat_id, user_id, current_persona)

        # Store image event (formatted to match your style)
        persona_ctx["message_history"].append({
            "text": f"[Image: {user_question}]",
            "sender": "user",
            "persona": None
        })

        # --- Your Original API Call (EXACTLY AS IS) ---
        persona_config = PERSONAS[current_persona]
        prompt_text = (
            f"{persona_config['system']}\n\n"
            f"Запрос: {user_question}\n\n"
            "Ответь в своём стиле (макс. 3 предложения)"
        )

        processing_msg = await update.message.reply_text("Разглядываю")
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # YOUR ORIGINAL MODEL - NOT CHANGED
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"  # Your original detail level
                        }
                    }
                ]
            }],
            max_tokens=250  # Your original token limit
        )

        # --- NEW: Store Bot Response ---
        analysis = response.choices[0].message.content
        persona_ctx["message_history"].append({
            "text": analysis,
            "sender": "bot",
            "persona": current_persona.value
        })

        await processing_msg.edit_text(analysis[:1000])  # Your original truncation

    except Exception as e:
        logger.error(f"Image error: {e}")
        await update.message.reply_text("Чёт не вышло. Попробуй другую картинку.")

async def group_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Check for either @mention OR reply-to-bot
    bot_username = context.bot.username.lower()
    is_reply_to_bot = (
            update.message.reply_to_message and
            update.message.reply_to_message.from_user.id == context.bot.id
    )
    has_mention = (
            (update.message.text and f"@{bot_username}" in update.message.text.lower()) or
            (update.message.caption and f"@{bot_username}" in update.message.caption.lower())
    )

    if not (is_reply_to_bot or has_mention):
        return  # Ignore normal group messages

    chat_id = update.message.chat.id
    user_id = update.effective_user.id


    # Route to appropriate handler
    if update.message.photo:
        await handle_image(update, context)
    elif update.message.document:
        await handle_file(update, context)
    else:
        # === ЕДИНСТВЕННОЕ ИЗМЕНЕНИЕ ===
        payload = build_prompt(
            chat_id=chat_id,
            user_input=update.message.text,
            persona_name=chat_modes[chat_id],
            user_id=user_id)

        response = await call_deepseek(payload)
        await update.message.reply_text(response)
    # --------------------------------------


# REGISTER ALL COMMANDS
# --------------------------------------
commands = [
    ("news", news),
    ("wtf", wtf),
    ("problem", problem),
    ("fugoff", fugoff),
    ("randomeme", randomeme),
    ("sych", sych),
    ("putin", putin),
    ("zhir", zhir),
    ("hohly", hohly),
    ("sage", sage),
    ("watts", watts),
    ("mode", set_mode),
    ("petros", petros)
]

for cmd, handler in commands:
    app.add_handler(CommandHandler(cmd, handler))

# ===== 1. PRIVATE CHAT HANDLER (keep) =====
app.add_handler(MessageHandler(
    filters.ChatType.PRIVATE & (filters.TEXT | filters.PHOTO | filters.Document.ALL),
    lambda update, ctx: (
        handle_image(update, ctx) if update.message.photo else
        handle_file(update, ctx) if update.message.document else
        handle_mention(update, ctx)
    )
))

# ===== 2. GROUP CHAT HANDLER (REPLACE with this) =====
app.add_handler(MessageHandler(
    filters.ChatType.GROUPS & (filters.TEXT | filters.PHOTO | filters.Document.ALL),
    group_handler  # Your custom function that checks @mentions
))

# ===== 3. REPLY HANDLER (keep one) =====
app.add_handler(MessageHandler(
    filters.TEXT & filters.REPLY,
    handle_reply
))

if __name__ == "__main__":
    print("⚡ Helper запущен с точным набором функций")
    app.run_polling()
