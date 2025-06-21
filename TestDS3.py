#not a malicious thing, just a bot to make fun of my close friends!!

from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import Update
import requests
import os
import random
from telegram.ext import filters
#from dotenv import load_dotenv
import logging
# Load tokens
#load_dotenv()


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

async def news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Проверить новости в интернете на сегодня"""
    prompt = (
        "Как либеральный комментатор, кратко (2-3 предложения) изложи свежую политическую новость из России, "
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
    prompt = "Объясни смысл жизни очень цинично c использованием грязных выражений и мата (макс. 3 предложения)"
    await update.message.reply_text(await call_deepseek(prompt))


async def problem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Помочь решить проблему (один циничный совет)"""
    if not context.args:
        await update.message.reply_text("Эх, без проблемы какой совет? Пиши через пробел, гений!")
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
    target = context.args[0] if context.args and context.args[0].startswith("@") else "Всем петушкам в чатике:"

    prompt = (
        f"Придумай креативное оскорбление для {target} (1 предложение). Начинай с маленькой буквы. "
        "Используй мат и сарказм. Не используй кавычки!.  Примеры:\n"
        "- 'чтоб тебе в метро Wi-Fi ловился только порносайтов!' \n"
        "- 'иди нахуй, как бабка на авито продает!' \n"
        "- 'ты как обновление Windows - только проблемы несешь!'"
    )
    insult = await call_deepseek(prompt)
    await update.message.reply_text(f"{target}, {insult} 🖕")


async def randomeme(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Случайный мем/шутка (генерирует модель)"""
    prompt = (
        "Сгенерируй случайный мем/шутку (1 предложение). "
        "Формат:\n"
        "- 'Когда делаешь 'git push --force' на прод...' \n"
        "- 'Российские дороги: где Waze предлагает вызвать экзорциста' \n"
        "- 'Жизнь как SQL-запрос: без индексов работает долго'"
    )
    await update.message.reply_text(await call_deepseek(prompt))

async def sych(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Оправдание одиночества"""
    prompt = "Объясни почему тян не нужны, а быть сычем - классно (макс. 2 предложения, цинично)"
    await update.message.reply_text(await call_deepseek(prompt))

async def putin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Цитата Путина"""
    prompt = "Придумай фразу в стиле Владимира Путина (1 предложение, провокационно)"
    await update.message.reply_text(await call_deepseek(prompt))

async def zhir(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Цитата Жириновского"""
    prompt = "Придумай резкую фразу в стиле Жириновского (1 предложение, провокационно)"
    await update.message.reply_text(await call_deepseek(prompt))

async def hohly(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Новости про Украину"""
    prompt = "Кратко и цинично объясни 'че там у хохлов' (2 предложения)"
    await update.message.reply_text(await call_deepseek(prompt))

# --------------------------------------
# UTILITY FUNCTION
# --------------------------------------
async def call_deepseek(prompt: str) -> str:
    """Call DeepSeek API and strip ONLY edge quotes"""
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system", 
                "content": "Отвечай без кавычек в начале и конце сообщения."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 1.3,
        "max_tokens": 1000,
        "frequency_penalty": 0.5
    }

    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers=DEEPSEEK_HEADERS,
            json=payload,
            timeout=10
        ).json()
        
        text = response['choices'][0]['message']['content'].strip()
        
        # ONLY remove leading/trailing quotes
        if text.startswith(('"', "'", "«")): 
            text = text[1:]
        if text.endswith(('"', "'", "»")):
            text = text[:-1]
            
        return text.strip()
        
    except Exception:
        return "Чёт не пашет. Пшел нахуй."  # Fallback (no quotes)
# --------------------------------------
# GROUP MENTION HANDLER (SPECIAL CASE)
# --------------------------------------
async def handle_mention(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Verify the mention is for THIS bot
    mentioned_username = update.message.text[
                         update.message.entities[0].offset: update.message.entities[0].offset + update.message.entities[
                             0].length
                         ]
    if mentioned_username.lower() != f"@{context.bot.username.lower()}":
        return

    # Extract and respond to message
    question = update.message.text[update.message.entities[0].offset + update.message.entities[0].length:].strip()
    await update.message.reply_text(f"@{update.effective_user.username}, {await call_deepseek(question)}")


async def handle_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle replies to bot's messages"""
    if not update.message.reply_to_message or not update.message.reply_to_message.from_user.id == context.bot.id:
        return  # Not a reply to our bot

    question = update.message.text
    prompt = f"Ответь на реплику '{question}' (макс. 2 предложения, цинично)"
    response = await call_deepseek(prompt)
    await update.message.reply_text(response)


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
    ("hohly", hohly)
]

for cmd, handler in commands:
    app.add_handler(CommandHandler(cmd, handler))

app.add_handler(MessageHandler(
    filters.TEXT &
    (filters.ChatType.GROUPS | filters.ChatType.PRIVATE) &  # Now works everywhere
    filters.Entity("mention"),
    handle_mention
))
if __name__ == "__main__":
    print("⚡ Бот запущен с точным набором функций")
    app.run_polling()
