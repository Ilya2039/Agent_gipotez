from __future__ import annotations
from typing import List, Optional

"""
🔥 Unified prompt builders for the strategy bot.
Фокус: креативные, но реалистичные гипотезы + умные вопросы по фактам клиента и протоколам встреч.
"""

def build_generate_discovery_question_prompt(
    qa_json: str,
    dialog_json: str = "",
    examples_text: str = "",
    avoid: Optional[List[str]] = None,
    count: int = 1,
    theme: str = "",
    avoid_unknown: Optional[List[str]] = None,
) -> str:
    context_block = (
        f"\nКонтекст (материалы клиента + протоколы встреч, укорочено, JSON):\n{dialog_json}\n"
        if dialog_json else ""
    )
    theme_block = f"\nТема анализа (если указана, фокусируйся на ней): {theme}\n" if theme else ""
    examples_block = (
        "\nПримеры типовых гипотез (для ориентира; НЕ выбирай их, формулируй свои вопросы):\n"
        + examples_text + "\n" if examples_text else ""
    )
    avoid_block = (
        ("\nНе повторяй и не перефразируй эти вопросы:\n- " + "\n- ".join(avoid)) if avoid else ""
    )
    unknown_block = (
        ("\nСмени тему и не уточняй эти аспекты — КМ ответил, что не знает:\n- " + "\n- ".join(avoid_unknown))
        if avoid_unknown else ""
    )
    return (
        "РОЛЬ: Старший интервьюер/консалтер.\n"
        + "ЦЕЛЬ: уточнить деятельность компании — продукты, процессы, клиентов, рынок, внутренние проблемы.\n"
        + "ИСТОЧНИКИ: сопоставляй материалы клиента и протоколы встреч; опирайся на сигналы из разговоров.\n"
        + "ТРЕБОВАНИЯ: вопрос прикладной, конкретный, ≤ 18 слов, заканчивается на '?'. Без общих слов и 'как вы планируете развивать'.\n"
        + "СТРОГИЙ ВЫВОД — только JSON-массив из ровно N вопросов.\n"
        + f"N={count}.\n"
        + f"История Q/A (JSON):\n{qa_json}\n"
        + f"{context_block}{theme_block}{examples_block}{avoid_block}{unknown_block}"
        + "Формат ответа: [\"Вопрос?\"]"
    )


def build_decide_next_action_prompt(
    theme: str,
    qa_json: str,
    dialog_json: str = "",
    prefer_non_finance: bool = True,
) -> str:
    context_block = (
        f"\nКонтекст (материалы клиента + протоколы встреч, укорочено, JSON):\n{dialog_json}\n"
        if dialog_json else ""
    )
    deprior = (
        "Финансовые гипотезы формируй только при наличии фактов о долгах, выручке, убытках или инвестициях.\n"
        if prefer_non_finance else ""
    )
    return (
        "РОЛЬ: Старший стратег-консультант с отраслевой экспертизой.\n"
        + "ЦЕЛЬ: принять решение — задать вопрос, запросить материалы, сформулировать гипотезу или завершить.\n"
        + "ФОКУС: реальные управленческие и рыночные вызовы клиента, а не абстрактные рассуждения.\n"
        + deprior
        + "КРИТЕРИИ:\n"
        + "- Недостаточно данных → action=ask.\n"
        + "- Не хватает цифр → action=request_file.\n"
        + "- Есть устойчивая картина → action=hypothesis.\n"
        + "- Противоречия между ответами и протоколами → уточни.\n"
        + "Формат вывода строго JSON:\n"
        + '{"action":"ask|request_file|hypothesis|stop","question":"<если ask>","note":"<почему>"}\n'
        + f"Тема: {theme}\nИстория Q/A (JSON):\n{qa_json}\n{context_block}JSON:"
    )


def build_generate_free_hypothesis_prompt(
    theme: str,
    qa_json: str,
    dialog_json: str = "",
    examples_text: str = "",
    prefer_non_finance: bool = True,
) -> str:
    context_block = (
        f"\nКонтекст (материалы клиента + протоколы встреч, укорочено, JSON):\n{dialog_json}\n"
        if dialog_json else ""
    )
    examples_block = (
        "\nПримеры типовых гипотез (для ориентира, не копируй):\n" + examples_text + "\n"
        if examples_text else ""
    )
    deprior = (
        "\nФинансовые гипотезы допускаются только при прямых сигналах о долгах, убытках или прибыли.\n"
        if prefer_non_finance else ""
    )
    return (
        "РОЛЬ: Ведущий стратег-консультант с креативным, но прагматичным мышлением.\n"
        + "ЦЕЛЬ: сформулировать ОДНУ гипотезу, которая соединяет факты клиента с новым, но реалистичным направлением.\n"
        + "ПОДХОД:\n"
        + "- Найди слабое место или неиспользованный потенциал компании по материалам и протоколам.\n"
        + "- Предложи неожиданный, но реализуемый рычаг: организационный, операционный, продуктовый, партнёрский или управленческий.\n"
        + "- Гипотеза должна быть применима в горизонте 6–18 месяцев.\n"
        + "- Избегай 'модных' слов (платформа, блокчейн, метавселенная, AI), если их нет в контексте.\n"
        + "- Креатив приветствуется: допускаются сценарные или парадоксальные углы ('а что если').\n"
        + deprior
        + "СТРОГИЙ ВЫВОД — только JSON:\n"
        + "{\n"
        + '  "hypothesis": "<лаконичная, стратегически насыщенная формулировка>",\n'
        + '  "reason": "<10–25 слов, какие сигналы или противоречия из контекста на неё указывают>",\n'
        + '  "tags": ["отрасль", "подотрасль", "ключевые процессы или факторы"]\n'
        + "}\n"
        + "КРИТЕРИИ: новизна, применимость, инсайт, связность с реальными фактами.\n"
        + f"Тема: {theme}\nИстория (Q/A JSON):\n{qa_json}\n{context_block}{examples_block}JSON:"
    )


def build_generate_alternative_hypothesis_prompt(
    qa_json: str,
    dialog_json: str = "",
    examples_text: str = "",
    avoid_hypothesis: str = "",
    prefer_non_finance: bool = True,
) -> str:
    context_block = (
        f"\nКонтекст (материалы клиента + протоколы встреч, укорочено, JSON):\n{dialog_json}\n"
        if dialog_json else ""
    )
    examples_block = (
        "\nПримеры типовых гипотез (для ориентира):\n" + examples_text + "\n"
        if examples_text else ""
    )
    avoid_block = f"\nИсключить гипотезу: {avoid_hypothesis}\n" if avoid_hypothesis else ""
    deprior = (
        "\nФинансовые гипотезы только если контекст явно подтверждает наличие долгов/убытков.\n"
        if prefer_non_finance else ""
    )
    return (
        "РОЛЬ: Стратег-консультант уровня партнёра, создающий реалистичную альтернативу.\n"
        + "ЦЕЛЬ: предложить АЛЬТЕРНАТИВНУЮ гипотезу, которая раскрывает другой, но правдоподобный сценарий действий.\n"
        + "ПОДХОД:\n"
        + "- Найди другой рычаг, не повторяющий исходную идею.\n"
        + "- Используй факты из протоколов: риски, решения, финансовые ограничения, намерения менеджмента.\n"
        + "- Идея должна быть реалистична для запуска в горизонте 12 месяцев.\n"
        + "- Можно быть креативным: неожиданный партнёр, нетривиальная бизнес-модель, новый сегмент.\n"
        + "- Не повторяй термины из avoid_hypothesis.\n"
        + deprior
        + "СТРОГИЙ ВЫВОД — только JSON:\n"
        + "{\n"
        + '  "hypothesis": "<чёткая, альтернативная формулировка>",\n'
        + '  "reason": "<10–25 слов, почему это направление перспективно и чем отличается от предыдущей гипотезы>",\n'
        + '  "tags": ["отрасль", "подотрасль", "ключевые процессы или факторы"]\n'
        + "}\n"
        + "КРИТЕРИИ: новизна, реалистичность, стратегическая ценность, креатив.\n"
        + f"История (Q/A JSON):\n{qa_json}\n{context_block}{examples_block}{avoid_block}JSON:"
    )


def build_generate_meeting_questions_prompt(
    hypothesis: str,
    qa_json: str,
    dialog_json: str = "",
    count: int = 4,
    avoid_questions: Optional[List[str]] = None,
) -> str:
    context_block = (
        f"\nКонтекст (материалы клиента + протоколы встреч, укорочено, JSON):\n{dialog_json}\n"
        if dialog_json else ""
    )
    avoid_block = (
        ("\nНе повторяй и не перефразируй эти вопросы:\n- " + "\n- ".join(avoid_questions))
        if avoid_questions else ""
    )
    return (
        "РОЛЬ: Партнёр, ведущий стратегическую встречу с клиентом.\n"
        + "ЦЕЛЬ: задать 3–4 точечных вопроса для проверки гипотезы и выявления рисков.\n"
        + "ТРЕБОВАНИЯ: прикладные, контекстные, ≤ 18 слов, без общих фраз.\n"
        + "Если вопрос уже освещён в протоколах, предложи следующее логичное уточнение.\n"
        + "СТРОГИЙ ВЫВОД — JSON-массив из N строк.\n"
        + f"N={count}.\n"
        + f"Гипотеза: {hypothesis}\nИстория Q/A (JSON):\n{qa_json}\n{context_block}{avoid_block}Формат: [\"Вопрос?\", ...]"
    )


def build_is_unknown_answer_prompt(answer: str) -> str:
    return (
        "Классифицируй ответ пользователя: есть ли признак 'не знаю/нет данных/затрудняюсь ответить'.\n"
        "Верни строго JSON: {\"unknown\": true|false}.\n"
        f"Ответ пользователя: {answer}\nJSON:"
    )


def build_example_answer_prompt(question: str, answers_json: str, dialog_json: str = "") -> str:
    dialog_block = (
        f"\nКонтекст (материалы клиента + протоколы встреч, укорочено, JSON):\n{dialog_json}\n"
        if dialog_json else ""
    )
    return (
        "Сформируй правдоподобный короткий ответ (одно предложение, ≤ 12 слов).\n"
        "Без лишних вводных и дисклеймеров. По делу, естественно, по-русски.\n"
        f"Вопрос: {question}\nОтветы КМ (JSON):\n{answers_json}\n{dialog_block}"
        "Верни только одно предложение без кавычек."
    )

