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

import sys

chat_memories = defaultdict(lambda: deque(maxlen=32))

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

async def set_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat.id
    if not context.args:
        await update.message.reply_text("Доступные режимы: normal, good, phil")
        return

    mode_name = context.args[0].lower()
    try:
        persona = Persona(mode_name)  # Пытаемся найти в enum
        chat_modes[chat_id] = persona.value  # Сохраняем строку (normal/volodia/etc)
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
        chat_history: deque = None
) -> dict:
    """
    Собирает payload для DeepSeek API.

    Args:
        chat_id: ID чата (для истории)
        user_input: Текст сообщения пользователя
        persona_name: Один из Persona (normal, volodia и т.д.)
        chat_history: Очередь с историей сообщений (если None - берет из chat_memories)

    Returns:
        Готовый payload для call_deepseek
    """
    # Получаем персонажа
    persona = PERSONAS.get(Persona(persona_name), PERSONAS[Persona.NORMAL])  # fallback на normal

    # Берем историю чата (если не передана явно)
    history = chat_history if chat_history is not None else chat_memories[chat_id]

    # Форматируем историю
    formatted_history = "\n".join(
        f"{i}. {msg}" for i, msg in enumerate(history, 1)
    ) if history else "Истории нет"

    # Собираем system-промпт
    system_prompt = persona["system"].format(
        chat_history=formatted_history,
        user_input=user_input
    )

    # Возвращаем готовый payload
    return {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
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

    chat_type = update.message.chat.type
    is_private = chat_type == "private"

    # Новое: проверка на реплай боту
    is_reply_to_bot = (
            not is_private
            and update.message.reply_to_message
            and update.message.reply_to_message.from_user.id == context.bot.id
    )

    # Обновлённое условие для групп
    if not is_private and not is_reply_to_bot:
        if not context.bot.username:
            return

        bot_username = context.bot.username.lower()
        message_text = update.message.text.lower()

        # Проверяем И @mention ИЛИ реплай
        if f"@{bot_username}" not in message_text.split():  # ← Сохраняем старую проверку
            return  # Но теперь is_reply_to_bot уже обработан выше

    # Существующая логика обработки (без изменений)
    chat_id = update.message.chat.id
    user_id = update.effective_user.id
    memory_key = (chat_id, user_id)

    if memory_key not in chat_memories:
        chat_memories[memory_key] = []

    chat_memories[memory_key].append(update.message.text)

    context_messages = "\n".join(chat_memories[memory_key])
    prompt = (
        f"Context (last messages):\n{context_messages}\n\n"
        f"New message: {update.message.text}\n\n"
        "Отвечай уверенно (макс. 3 предложения)"
    )

    payload = build_prompt(chat_id, prompt, chat_modes[chat_id])
    response = await call_deepseek(payload)
    await update.message.reply_text(response)


async def handle_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.reply_to_message.from_user.id == context.bot.id:
        return

    chat_id = update.message.chat.id
    user_id = update.effective_user.id
    memory_key = (chat_id, user_id)

    chat_memories[memory_key].append(update.message.text)

    context_messages = "\n".join(chat_memories[memory_key])
    prompt = f"Context:\n{context_messages}\n\nReply to: {update.message.text}"

    payload = build_prompt(chat_id, prompt, chat_modes[chat_id])
    response = await call_deepseek(payload)
    await update.message.reply_text(response)


async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # IMMEDIATELY skip all photo-type documents
    # if update.message.photo or (update.message.document and update.message.document.mime_type.startswith('image/')):
    #     return
    if update.message.photo or (update.message.document and update.message.document.mime_type.startswith('image/')):
        await handle_image(update, context)  # Send to image handler
        return  # Exit to avoid double-processing

    if os.name == 'nt':  # Только для Windows
        os.makedirs('/tmp/', exist_ok=True)

    """Process ONLY PDF/DOCX/TXT/CSV files (strictly ignores images)"""
    # 1. Early exit for non-documents or images
    if not update.message.document:
        return  # Let other handlers process it

    if update.message.photo:
        return  # handle_image() will catch this

    if (update.message.document
            and update.message.document.mime_type.startswith('image/')
            and not update.message.photo):
        await handle_image(update, context)  # Force-process as image
        return

    # 2. Validate file type
    ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt", ".csv"]
    try:
        file_ext = os.path.splitext(update.message.document.file_name)[1].lower()
    except AttributeError:
        await update.message.reply_text("❌ Не могу определить тип файла.")
        return

    if file_ext not in ALLOWED_EXTENSIONS:
        await update.message.reply_text("❌ Только PDF/DOCX/TXT/CSV.")
        return

    # 3. Download and process
    progress_msg = await update.message.reply_text("📥 Загружаю файл...")
    file_path = f"/tmp/{int(time.time())}_{update.message.document.file_name}"

    try:
        # NEW: Retry download up to 3 times
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

        # NEW: Validate DOCX structure before processing
        if file_ext == ".docx":
            from zipfile import ZipFile
            with ZipFile(file_path) as z:
                if 'word/document.xml' not in z.namelist():
                    await progress_msg.edit_text("❌ Файл DOCX повреждён.")
                    return
        # Parse content
        text = ""
        if file_ext == ".pdf":
            # from pdfminer.high_level import extract_text
            # text = extract_text(file_path)
            # if not text.strip():
            #     await progress_msg.edit_text("🤷‍♂️ PDF пустой или только картинки.")
            #     return

            from pdfminer.high_level import extract_text
            try:
                text = extract_text(file_path)  # Let it fail naturally
            except Exception as e:
                logger.error(f"PDF Error: {e}")
                await progress_msg.edit_text("🤖 Не смог прочитать PDF (попробуй другой файл)")
                return
        elif file_ext == ".docx":
            from docx import Document
            doc = Document(file_path)
            text = "\n".join(p.text for p in doc.paragraphs if p.text)
        elif file_ext in (".txt", ".csv"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read(3000)  # Limit read for large files

        if not text.strip():
            await progress_msg.edit_text("🤷‍♂️ Файл пустой или нечитаемый.")
            return

        payload = build_prompt(
            chat_id=update.message.chat.id,
            user_input=f"Резюме документа (3 предложения):\n{text}",
            persona_name=chat_modes[update.message.chat.id]
        )
        summary = await call_deepseek(payload)
        await progress_msg.edit_text(f"📄 Вывод:\n{summary[:1000]}")  # Truncate long output

    except Exception as e:
        logger.error(f"FILE ERROR: {str(e)}", exc_info=True)
        await progress_msg.edit_text("💥 Ошибка обработки. Попробуй другой файл.")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle images with custom user prompts"""
    if not update.message.photo:
        await update.message.reply_text("Это не изображение.")
        return

    try:
        # Get image
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = BytesIO()
        await photo_file.download_to_memory(out=photo_bytes)
        base64_image = base64.b64encode(photo_bytes.getvalue()).decode('utf-8')

        # Get user's question
        user_question = (
                update.message.caption or
                (update.message.reply_to_message.text if update.message.reply_to_message else None) or
                "Опиши это изображение. "  # Default
        )

        # Build prompt
        persona = Persona(chat_modes[update.message.chat.id])  # Получаем текущую персону
        persona_config = PERSONAS[persona]  # Берем её конфиг

        prompt_text = (
            f"{persona_config['system']}\n\n"  # Описание стиля персоны
            f"Запрос: {user_question}\n\n"
            "Ответь в своём стиле (макс. 3 предложения)"
        )
        # Process
        processing_msg = await update.message.reply_text("Разглядываю")
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"
                        }
                    }
                ]
            }],
            max_tokens=250
        )

        # Send result
        analysis = response.choices[0].message.content
        await processing_msg.edit_text(analysis[:1000])

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

    # Route to appropriate handler
    if update.message.photo:
        await handle_image(update, context)
    elif update.message.document:
        await handle_file(update, context)
    else:
        # === ЕДИНСТВЕННОЕ ИЗМЕНЕНИЕ ===
        payload = build_prompt(
            chat_id=update.message.chat.id,
            user_input=update.message.text,
            persona_name=chat_modes[update.message.chat.id]
        )
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
