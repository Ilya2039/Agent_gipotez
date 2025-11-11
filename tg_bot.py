import os
import yaml
import asyncio
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.filters import Command
from app.core.pipeline import ClientPipeline
from app.core.logger import log


with open("cfg/messages.yaml", "r") as f:
    messages_cfg = yaml.safe_load(f)

with open("cfg/common.yaml", "r") as f:
    common_cfg = yaml.safe_load(f)

with open("cfg/prompts.yaml", "r") as f:
    prompts_cfg = yaml.safe_load(f)

load_dotenv()

API_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

user_db = {}
debounce_tasks = {}
question_index = {}  # Отслеживаем индекс текущего вопроса для каждого пользователя
bot = Bot(token=API_TOKEN)
dp = Dispatcher()


class Scenario(StatesGroup):
    docs_and_summary = State()
    ask_questions = State()
    answering_questions = State()
    make_hypothesis = State()


@dp.message(Command("start"))
async def cmd_start(message: types.Message, state: FSMContext):
    await message.answer(messages_cfg["upload_step"]["prompt"])
    await state.set_state(Scenario.docs_and_summary)


@dp.message(Scenario.docs_and_summary, F.content_type == "document")
async def process_docs(message: types.Message, state: FSMContext):
    """Принимает .docx (и .json) файлы, сохраняет в upload/<chat_id>/
    и запускает обработку.
    """
    chat_id = message.chat.id
    doc = message.document
    if not doc.file_name.endswith('.docx'):
        await message.reply("Поддерживаются только файлы .docx")

    folder = common_cfg['paths']['upload_dir'] + f'{chat_id}/'
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, doc.file_name)

    try:
        await bot.download(doc.file_id, destination=save_path)
        log.info(f"Файл {doc.file_name} сохранён: {save_path}")
    except Exception as e:
        log.error(f"Ошибка при сохранении файла: {e}")
        await message.reply("Не удалось скачать файл. Попробуйте ещё раз.")
        return

    # Создаём или получаем пайплайн для чата
    pipeline = user_db.get(chat_id)
    if pipeline is None:
        pipeline = ClientPipeline()
        user_db[chat_id] = pipeline

    if chat_id in debounce_tasks:
        debounce_tasks[chat_id].cancel()
    debounce_tasks[chat_id] = asyncio.create_task(debounce_process_docs(chat_id, state))


async def debounce_process_docs(chat_id, state, delay=common_cfg['debounce_time']):
    try:
        await asyncio.sleep(delay)
    except asyncio.CancelledError:
        return
    pipeline = user_db[chat_id]

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, pipeline.process_docx, common_cfg['paths']['upload_dir'] + f'{chat_id}/')
    await loop.run_in_executor(None, pipeline.find_arrangements, 'all')
    log.info(f"Arrangements после вызова all: {pipeline.arrangements}")
    await loop.run_in_executor(None, pipeline.find_arrangements, 'main')
    log.info(f"Arrangements после вызова main: {pipeline.arrangements}")

    del debounce_tasks[chat_id]
    log.info(f'Договоренности: {pipeline.arrangements}')
    result_all = f"""
{messages_cfg['arrangements']['prompt']}
{pipeline.arrangements.get('all', '')}
    """

    result_main = f"""
    {messages_cfg['arrangements']['main_points']}
{pipeline.arrangements.get('main', '')}
{messages_cfg['arrangements']['question']}
    """

    await bot.send_message(chat_id, result_all)
    await bot.send_message(chat_id, result_main)

    # переход к следующему шагу
    await state.set_state(Scenario.ask_questions)


@dp.message(Scenario.ask_questions)
async def ask_questions_step(message: types.Message, state: FSMContext):
    chat_id = message.chat.id
    pipeline = user_db.get(chat_id)

    pipeline.additional_context = f"""
{pipeline.additional_context}
{messages_cfg['arrangements']['question']}
{message.text}
    """
    loop = asyncio.get_event_loop()
    await message.reply(messages_cfg['analysis']['prompt'])
    await loop.run_in_executor(None, pipeline.generate_questions)
    log.info(f'Вопросы: {pipeline.questions}')
    
    # Инициализируем индекс вопроса
    question_index[chat_id] = 0
    
    # Переходим к состоянию ответов на вопросы
    await state.set_state(Scenario.answering_questions)
    
    # Отправляем первый вопрос
    if pipeline.questions:
        first_question = pipeline.questions[0]['question']
        q_text = (
            f"Вопрос 1/{len(pipeline.questions)}:\n{first_question}"
        )
        await message.answer(q_text)
    else:
        await message.answer("Не удалось сформировать вопросы.")
        await state.set_state(Scenario.make_hypothesis)


@dp.message(Scenario.answering_questions)
async def answer_question_step(message: types.Message, state: FSMContext):
    chat_id = message.chat.id
    pipeline = user_db.get(chat_id)
    idx = question_index.get(chat_id, 0)
    
    # Сохраняем ответ в additional_context
    current_question = pipeline.questions[idx]['question']
    answer_text = (
        f"\nВопрос: {current_question}\nОтвет: {message.text}"
    )
    pipeline.additional_context += answer_text
    
    # Переходим к следующему вопросу
    idx += 1
    question_index[chat_id] = idx
    
    # Если есть ещё вопросы
    if idx < len(pipeline.questions):
        next_question = pipeline.questions[idx]['question']
        await message.answer(
            f"Вопрос {idx + 1}/{len(pipeline.questions)}:\n{next_question}"
        )
    else:
        # Все вопросы отвечены, переходим к гипотезам
        await message.answer("Спасибо за ответы! Формирую гипотезы...")
        await state.set_state(Scenario.make_hypothesis)
        
        # Генерируем гипотезы
        loop = asyncio.get_event_loop()
        hypotheses = await loop.run_in_executor(
            None, pipeline.make_hypothesis
        )
        log.info(f'Гипотезы: {hypotheses}')
        
        # Отправляем каждую гипотезу отдельным сообщением
        if isinstance(hypotheses, list):
            for i, hyp in enumerate(hypotheses[:3], 1):
                if isinstance(hyp, dict):
                    hyp_text = hyp.get('hypothesis', str(hyp))
                else:
                    hyp_text = str(hyp)
                await message.answer(f"Гипотеза {i}:\n{hyp_text}")
        else:
            await message.answer(f"Гипотеза 1:\n{hypotheses}")

if __name__ == "__main__":
    asyncio.run(dp.start_polling(bot))
