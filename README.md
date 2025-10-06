## Agent Gipotez (Telegram bot)

Коротко
- Бот на aiogram 3.
- 6 фиксированных вопросов + 4 адаптивных уточняющих.
- Выбор основной гипотезы (LLM) + автоселект побочной и вывод её вопросов.
- DOCX парсится в JSON; финальная карточка гипотезы берётся из `data/hypotheses_cards.json` (23 карты) или режется из DOCX.
- Логи в `logs/`, промпт выбора гипотезы сохраняется в `logs/prompts/`.

Структура
- `app/bot/bot_app.py` — вход и хэндлеры.
- `app/prompts/prompts.py` — промпты (в т.ч. для побочной гипотезы).
- `app/llm/client.py` — GigaChat через LangChain (температура по умолчанию 0.0).
- `app/survey/model.py` — 6 вопросов, мультивыбор с маркерами.
- `app/data/hypotheses_repo.py` — карточки гипотез из DOCX/JSON.
- `app/data/sub_repo.py` — база побочных гипотез `data/subhypotheses.json`.
- `app/parsers/docx_parser.py` — парсер DOCX → JSON.

Подготовка
1) Python 3.12+, создать venv и установить зависимости:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
2) Создать `.env` в корне:
```env
TELEGRAM_BOT_TOKEN=xxxxx
GIGACHAT_AUTH=xxxxx
GIGACHAT_MODEL=GigaChat-2-Max
GIGACHAT_TEMPERATURE=0.0
GIGACHAT_TIMEOUT=60
GIGACHAT_PROFANITY_CHECK=false
```

Запуск
```bash
python main.py
```

Файлы данных
- `data/hypotheses.txt` — 23 основные гипотезы (по строке).
- `data/SP_Порядок_проработки_развилок_и_гипотез_2025_10_09.docx` — источник карточек.
- `data/hypotheses_cards.json` — 23 заранее вырезанные карточки (если есть — используется в первую очередь).
- `data/subhypotheses.json` — побочные гипотезы и вопросы.

Пуш в GitHub
(выполнить в корне репо)
```bash
git init
git remote add origin git@github.com:Ilya2039/Agent_gipotez.git
git add .
git commit -m "init bot with survey, LLM, cards and subhypotheses"
git branch -M main
git push -u origin main
```
