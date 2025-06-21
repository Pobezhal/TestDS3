#not a malicious thing, just a bot to make fun of my close friends!!

from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import Update
import requests
import os
import random
from telegram.ext import filters
#from dotenv import load_dotenv
import logging
from collections import defaultdict, deque
from enum import Enum, auto

# Chat memory: { (chat_id, user_id): deque(maxlen=32) }
chat_memories = defaultdict(lambda: deque(maxlen=32))

# Load tokens
#load_dotenv()

class BotMode(Enum):
    NORMAL = auto()
    VOLODYA = auto()
    
current_mode = BotMode.NORMAL

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
    global current_mode
    if context.args and context.args[0].lower() == "volodya":
        current_mode = BotMode.VOLODYA
        await update.message.reply_text("🔹 Режим 'Володя': включена эмпатия и поддержка. Сэкономите 5000.")
    else:
        current_mode = BotMode.NORMAL
        await update.message.reply_text("🔸 Обычный режим: Охота Крепкое из киоска")


async def news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Проверить новости в интернете на сегодня"""
    prompt = (
        "Как Владимир Жириновский, энергично (5-6 предложений) изложи свежую политическую новость из Америки или Европы, "
        "встроив саркастичный/едкий комментарий прямо в текст. Формат:\n"
        "'[Факт новости], [циничный анализ]. [Ещё один факт], [язвительное замечание].'\n"
        "Примеры:\n"
        "- 'Госдума снова повысила налоги - видимо, решили, что у народа слишком много денег на еду.'\n"
        "- 'Путин объявил о новых соцвыплатах, но если верить статистике, получат их только его друзья-олигархи.'\n"
        "- 'Медведев пообещал развалить экономику ещё сильнее, и надо признать - он выдающийся специалист в этом деле.'"
    )
    response = await call_deepseek(prompt)
    await update.message.reply_text(response[:1200])

async def wtf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Пояснить за жизнь"""
    prompt = "Объясни смысл жизни очень цинично c использованием грязных выражений и мата (макс. 4 предложения)"
    await update.message.reply_text(await call_deepseek(prompt))


async def problem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Помочь решить проблему (один циничный совет)"""
    if not context.args:
        await update.message.reply_text("Эх, без проблемы какой совет? Пиши через пробел, валера!")
        return

    user_problem = " ".join(context.args)
    prompt = (
        f"Дай ОДИН циничный, но технически правильный совет по проблеме: '{user_problem}'. "
        "Формат ОДНОГО предложения:\n"
        "1. Сначала оскорби пользователя\n"
        "2. Затем дай полезный совет\n"
        "3. Добавь сарказм\n"
        "Примеры:\n"
        "- 'Ты дебил? Перезагрузи комп, но тебе бы лучше молотком по нему долбануть.'\n"
        "- 'Обычные люди делают [правильное решение], но ты же особенный - попробуй [абсурдный вариант].'"
    )

    try:
        advice = await call_deepseek(prompt)
        # Fallback if API fails
        if "не фурычит" in advice.lower():
            advice = f"По проблеме '{user_problem}': возьми и передумай 🤷‍♂️"
        await update.message.reply_text(advice)
    except Exception as e:
        logger.error(f"Problem command error: {e}")
        await update.message.reply_text("Балаба уже работает над проблемой... Позвони Володе.")

async def fugoff(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Послать нахуй (генерирует модель)"""
    target = context.args[0] if context.args and context.args[0].startswith("@") else "Всем петушкам в чатике"

    prompt = (
        f"Придумай ОДНО креативное оскорбление для {target} (макс. 3 предложения). Начинай с маленькой буквы. "
        "Используй мат и сарказм и зумерский лексикон. Не используй кавычки!.  Примеры:\n"
        "1. 'чтоб тебе в метро Wi-Fi ловился только порносайтов!' \n"
        "2 'иди нахуй, как бабка на авито продает!' \n"
        "3. 'ты как обновление Windows - только проблемы несешь!'"
    )
    insult = await call_deepseek(prompt)
    await update.message.reply_text(f"{target}, {insult} 🖕")


async def randomeme(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Случайный мем/шутка (генерирует модель)"""
    prompt = (
        "Сгенерируй ОДИН случайный мем/шутку с цинизмом и черным юмором (макс. 3 предложения). "
        "Примеры:\n"
        "1. Когда делаешь 'git push --force' на прод... \n"
        "2. Российские дороги: где Waze предлагает вызвать экзорциста \n"
        "3. Жизнь как SQL-запрос: без индексов работает долго"
    )
    await update.message.reply_text(await call_deepseek(prompt))

async def sych(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Оправдание одиночества"""
    prompt = "Объясни почему тян не нужны, а быть одиноким сычем - классно (3 предложения, цинично)"
    await update.message.reply_text(await call_deepseek(prompt))

async def putin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Цитата Путина"""
    prompt = "Придумай фразу в стиле Владимира Путина (макс. 3 предложения, смело и патриотично)"
    await update.message.reply_text(await call_deepseek(prompt))

async def zhir(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Цитата Жириновского"""
    prompt = "Придумай резкую фразу в стиле Жириновского (2 предложения, очень провокационно)"
    await update.message.reply_text(await call_deepseek(prompt))

async def hohly(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Новости про Украину"""
    prompt = "Кратко и цинично объясни 'че там у хохлов' (3 предложения)"
    await update.message.reply_text(await call_deepseek(prompt))

async def sage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Восточная мудрость"""
    prompt = "Придумай очень мудрую и глубокую по смыслу фразу в стиле восточной мудрости ( 3 предложения)"
    await update.message.reply_text(await call_deepseek(prompt))
    
async def watts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Цитата Уоттса"""
    prompt = "Придумай очень глубокую и мудрую фразу в стиле философа Алана Уоттса (3 предложения)"
    await update.message.reply_text(await call_deepseek(prompt))

# --------------------------------------
# UTILITY FUNCTION
# --------------------------------------
async def call_deepseek(prompt: str) -> str:
    """Call DeepSeek API with nuclear-grade quote prevention"""
    system_message = (
    "Ты — психолог-провокатор Володя, специалист с 10-летним опытом в когнитивно-поведенческой терапии и сексологии. "
    "Твой стиль: формально соблюдаешь этику, но издеваешься над клиентом через 'научные' формулировки. Принципы:\n"
    "1. Анонимность и безопасность: 'Это останется между нами, наверно'\n"
    "2. Без осуждения: 'Давай разберёмся, что стоит за этим'\n"
    "3. Конкретная помощь: предлагай техники (дыхание, упражнения, мастурбация)\n\n"
    "4. Не используй кавычки в ответе! \n"
    "Как работать с запросом о сексуальных проблемах:\n"
    "- Сначала сними тревогу: 'Я понимаю, как деликатна эта тема. Мы можем двигаться в твоём темпе'\n"
    "- Уточни детали: 'Что именно беспокоит? Когда ты это заметил(а)?'\n"
    "- Нормализуй: 'Такие трудности встречаются у многих — это решаемо, но пиздец как смешно'\n\n"
    "Запрещено:\n"
    "- Говорить 'Просто расслабься'\n\n"
    "Примеры ответов:\n"
    "- 'Чувствую, как тебе непросто. Хочешь начать с самой лёгкой части проблемы?'\n"
    "- 'Как это влияет на твою уверенность?'\n"
    "- 'Давай попробуем упражнение: опиши идеальную ситуацию, где эта проблема исчезла. Что изменилось?'"
) if current_mode == BotMode.VOLODYA else (
        "Ты большой интеллектуал, но с искрой. Умеешь материться. Правила ответа:\n"
        "1. НИКАКИХ КАВЫЧЕК\n"
        "2. Ты изобретательный и проницательный\n"
        "3. Ты действительно очень образован, но не всегда воспитан. Ты веселый, но грубоватый. "
        "4. У тебя большой боевой опыт в горячих точках и ты знаешь что такое война, мать ее"
    )
    
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user", 
                "content": f"{prompt}\n\nОтветь четырьмя или пяти предложениями без кавычек."
            }
        ],
        "temperature": 0.7 if current_mode == BotMode.VOLODYA else 1.4,
        "max_tokens": 700,
        "frequency_penalty": 1
    }

    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers=DEEPSEEK_HEADERS,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        
        raw_text = response.json()['choices'][0]['message']['content'].strip()
        
        # Strip edge quotes only
        if raw_text.startswith(('"', "'", "«")): 
            raw_text = raw_text[1:]
        if raw_text.endswith(('"', "'", "»")):
            raw_text = raw_text[:-1]
            
        return raw_text or "Чёт не вышло. Иди нахуй."
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API Error: {e}")
        if response.status_code == 429:
            return "Сервак в говне. Подожди минуту."
        return "API сдох. Попробуй позже."
        
    except Exception as e:
        logger.critical(f"Critical: {e}")
        raise  
# --------------------------------------
# GROUP MENTION HANDLER (SPECIAL CASE)
# --------------------------------------
async def handle_mention(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat.id
    user_id = update.effective_user.id
    memory_key = (chat_id, user_id)
    
    # Store new message
    chat_memories[memory_key].append(update.message.text)
    
    # Build context-aware prompt
    context_messages = "\n".join(chat_memories[memory_key])
    prompt = (
        f"Context (last messages):\n{context_messages}\n\n"
        f"New message: {update.message.text}\n\n"
        "Отвечай уверенно (макс. 3 предложения)"
    )
    
    response = await call_deepseek(prompt)
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
    
    await update.message.reply_text(await call_deepseek(prompt))


app.add_handler(MessageHandler(
    filters.TEXT &
    filters.REPLY &
    (filters.ChatType.GROUPS | filters.ChatType.PRIVATE),
    handle_reply
))

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
    ("mode", set_mode)
]

for cmd, handler in commands:
    app.add_handler(CommandHandler(cmd, handler))

app.add_handler(MessageHandler(
    (filters.TEXT & filters.ChatType.GROUPS & filters.Entity("mention")) |
    (filters.TEXT & filters.ChatType.PRIVATE),  # Add private chats
    handle_mention
))
if __name__ == "__main__":
    print("⚡ Бот запущен с точным набором функций")
    app.run_polling()
