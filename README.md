## Hypothesis Agent (REST API)

Агент собирает контекст (файлы клиента + протоколы), задаёт 1–5 уточняющих вопросов, а затем формирует 3 гипотезы и 5 вопросов к встрече. Все действия логируются.

Технологии: FastAPI, LangGraph, LangChain (GigaChat), python‑docx.

### Используемые файлы (по делу)
- `main.py` — точка входа; запускает сервер и настраивает логи в консоль и `logs/bot.log`.
- `app/api/server.py` — сами эндпоинты и сессии; пишет посессионные логи в `logs/dialogs/<session_id>.log`.
- `app/engine/graph.py` — логика агента: выбор next‑action (decide/ask) и прямой finalize (3 гипотезы + 5 вопросов).
- `app/prompts/core.py` — все промпты (вопросы, выбор действия, гипотезы и альтернатива, вопросы к встрече, «не знаю», пример ответа).
- `app/services/examples.py` — генерация короткого примера ответа под вопросом.
- `app/services/unknowns.py` — детектор ответов «не знаю».
- `app/llm/client.py` — клиент GigaChat (конфиг из `.env`).
- `app/parsers/docx_parser.py` — парсер DOCX → JSON для контекста.
- `data/hypotheses.txt` — референс‑список гипотез (для ориентира модели).

### Переменные окружения (.env)
```env
GIGACHAT_AUTH=xxxxx
GIGACHAT_MODEL=GigaChat-2-Reasoning
GIGACHAT_TEMPERATURE=0.8
GIGACHAT_TIMEOUT=60
GIGACHAT_PROFANITY_CHECK=false
```

### Запуск
```bash
python main.py
```
Swagger UI: `http://localhost:8000/docs`

### Флоу в Swagger UI
1) `POST /session/start`
   - `session_id`: любой uid; `theme`: можно пусто. Если пусто — первый `qa/next` спросит тему.

2) `POST /session/upload`
   - Загружайте ПО ОДНОМУ файлу: выбрали → Execute. Повторите для всех файлов с тем же `session_id`.
   - В ответе поле `files` — сколько всего файлов уже учтено. В `logs/bot.log` будет `[upload] ... total=N`, а в `logs/dialogs/<session>.log` — `[FILE] <имя>`.

3) Вопросы
   - `POST /qa/next` → `{question, example, idx}` (сначала вопрос про тему, затем уточняющие, максимум 5).
   - `POST /qa/answer` → `{next: "ask" | "finalize"}`. Если `finalize` — переходите к финалу.

4) Финализация
   - `POST /finalize` → `{ "hypotheses": [{hypothesis, reason} x3], "meeting_questions": [x5], "unknowns": [xN] }`.
   - `unknowns` — список вопросов, где вы отвечали «не знаю» (покажите блоком «Будет полезно узнать у клиента»).

### Логи
- Общий лог: `logs/bot.log` (включая `[upload] ... total=N`).
- Диалоговый лог: `logs/dialogs/<session_id>.log` — вся канва беседы (START/Theme, FILE, Qn/EXAMPLE, An/ACT, FINAL, UNKNOWN).

### Траблшут
- 409 в `/qa/next` — достигнут лимит 5 вопросов → вызывайте `/finalize`.
- Пустой `/finalize` — повторите; есть фолбэк. Смотрите `[finalize]` в логах.
- Нет поддержки форм — установите `python-multipart`.
