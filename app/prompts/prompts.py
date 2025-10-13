from __future__ import annotations

from typing import List, Optional


def build_generate_followups_prompt(
    answers_json: str,
    count: int = 4,
    avoid: Optional[List[str]] = None,
    context: str = "",
    candidates_text: str = "",
) -> str:
    avoid_block = ""
    if avoid:
        avoid_joined = "\n- " + "\n- ".join(avoid)
        avoid_block = f"\nНе повторяй и не перефразируй эти вопросы:{avoid_joined}\n"
    context_block = f"\nДополнительный контекст:\n{context}\n" if context else ""
    candidates_block = (
        f"\nТекущий пул наиболее вероятных гипотез (для уточнения и разведения):\n{candidates_text}\n"
        if candidates_text
        else ""
    )
    return (
        "Ты — профессиональный консалтер. Сформируй уточняющие вопросы для дальнейшей диагностики возможностей развития и преодоления трудностей бизнеса клиента.\n"
        "Цель вопросов — помочь разобраться в ситуации и сфокусировать анализ; это не тест и не проверка знаний.\n"
        "Формулируй по-человечески, естественно и доброжелательно.\n"
        "Если информации по пункту не было в файле/ответах, можно использовать форму: \"Я в файле не нашёл(а) информацию о <X>, подскажите, какая она?\"\n"
        "Каждый следующий вопрос адаптируй с учётом предыдущих ответов (не задавай то, на что уже отвечено).\n" \
        "ЗАПРЕЩЕНО задавать вопросы о финансовых показателях холдинга или клиента."
        "СТРОГОЕ ТРЕБОВАНИЕ ВЫВОДА: верни только JSON-массив из ровно N строк без какого-либо текста вокруг.\n"
        "Каждый элемент массива — одна строка-вопрос на русском, ≤ 18 слов, обязательно оканчивается на '?'.\n"
        "Запрещены дисклеймеры, вводные, пояснения, комментарии, списки, нумерации, markdown. Только JSON-массив.\n"
        f"N={count}.\n"
        f"Ответы клиентского менеджера (JSON):\n{answers_json}\n"
        f"{context_block}"
        f"{candidates_block}"
        f"{avoid_block}"
        "Верни итог в формате: [\"Вопрос 1?\", \"Вопрос 2?\", \"Вопрос 3?\", \"Вопрос 4?\"]"
    )


def build_example_answer_prompt(question: str, answers_json: str, dialog_json: str = "") -> str:
    dialog_block = f"\nКонтекст (из файла, укорочен):\n{dialog_json}\n" if dialog_json else ""
    return (
        "Сформируй короткий пример ответа пользователя на следующий вопрос ровно в ОДНОМ предложении (не более 12 слов), по-русски.\n"
        "Ответ должен выглядеть как правдоподобная реплика по делу, без дисклеймеров и пояснений. Избегай сложных перечислений.\n"
        f"Вопрос: {question}\n"
        f"Ответы клиентского менеджера (JSON):\n{answers_json}\n"
        f"{dialog_block}"
        "Верни только ОДНО предложение (≤ 12 слов) без кавычек."
    )


def build_select_hypothesis_prompt(
    answers_json: str,
    hypotheses_text: str,
    dialog_json: str = "",
    facts_summary: str = "",
) -> str:
    context_block = f"\nКонтекст (JSON из файла, укорочен):\n{dialog_json}\n" if dialog_json else ""
    facts_block = f"\nФакты (обязательно учесть):\n{facts_summary}\n" if facts_summary else ""
    return (
        "Ты — профессиональный косалтер. На входе ответы клиентского менеджера о бизнесе клиента и список типовых гипотез (по одной в строке).\n"
        "ЗАДАЧА: выбрать ОДНУ гипотезу из списка, которая ЛУЧШЕ ВСЕГО СООТВЕТСТВУЕТ фактам о клиенте.\n"
        "ОГРАНИЧЕНИЯ: если факты противоречат формулировке гипотезы (например, выручка падает, а в гипотезе сказано, что она растёт/стабильна) — такую гипотезу выбирать НЕЛЬЗЯ.\n"
        "Учитывай тренды отрасли, её специфику, стагнацию/падение рынка, текучесть кадров, ликвидность, действия конкурентов.\n"
        "СТРОГИЙ ВЫВОД: верни только JSON-объект без текста вокруг: "
        '{"hypothesis":"<точная строка из списка>", "reason":"<опора на факты, 10-20 слов>"}'
        f"\nОтветы клиентского менеджера (JSON):\n{answers_json}\n"
        f"{context_block}"
        f"{facts_block}"
        f"Список гипотез (по одной в строке):\n{hypotheses_text}\n\n"
        "JSON:"
    )


def build_validate_hypothesis_prompt(answers_json: str, hypothesis: str, dialog_json: str = "", facts_summary: str = "") -> str:
    context_block = f"\nКонтекст (JSON из файла, укорочен):\n{dialog_json}\n" if dialog_json else ""
    facts_block = f"\nФакты:\n{facts_summary}\n" if facts_summary else ""
    return (
        "Проверь соответствие гипотезы фактам из ответов. Верни только JSON: "
        '{"valid": true|false, "conflicts": ["краткая причина", ...]} .\n'
        "Если в фактах указано падение выручки/прибыли, а гипотеза предполагает рост/стабильность — это конфликт.\n"
        f"Ответы клиентского менеджера (JSON):\n{answers_json}\n"
        f"{context_block}"
        f"{facts_block}"
        f"Гипотеза: {hypothesis}\n"
        "JSON:"
    )


def build_select_hypothesis_alternative(
    answers_json: str,
    hypotheses_text: str,
    avoid: str,
    dialog_json: str = "",
    facts_summary: str = "",
) -> str:
    context_block = f"\nКонтекст (JSON из файла, укорочен):\n{dialog_json}\n" if dialog_json else ""
    facts_block = f"\nФакты:\n{facts_summary}\n" if facts_summary else ""
    return (
        "Выбери ЛУЧШУЮ альтернативную гипотезу из списка, исключив указанную. Верни только JSON: "
        '{"hypothesis":"<точная строка из списка>", "reason":"<опора на факты>"}'
        f"\nОтветы клиентского менеджера (JSON):\n{answers_json}\n"
        f"{context_block}"
        f"{facts_block}"
        f"Исключить: {avoid}\n"
        f"Список гипотез:\n{hypotheses_text}\n\n"
        "JSON:"
    )


def build_select_subhypothesis_prompt(
    answers_json: str,
    main_hypothesis: str,
    sub_options_text: str,
    dialog_json: str = "",
    facts_summary: str = "",
) -> str:
    context_block = f"\nКонтекст (JSON из файла, укорочен):\n{dialog_json}\n" if dialog_json else ""
    facts_block = f"\nФакты:\n{facts_summary}\n" if facts_summary else ""
    return (
        "Ты — профессиональный консалтер. На входе ответы клиентского менеджера и список побочных гипотез для выбранной основной. Гипотеза - ситуация, которая подходит к конкретному клиенту.\n"
        "ЗАДАЧА: выбрать ОДНУ побочную гипотезу, которая лучше всего соответствует фактам.\n"
        "СТРОГИЙ ВЫВОД: верни только JSON без текста: {\"sub\": \"<точная строка из списка>\"}.\n"
        f"Основная гипотеза: {main_hypothesis}\n"
        f"Ответы клиентского менеджера (JSON):\n{answers_json}\n"
        f"{context_block}"
        f"{facts_block}"
        f"Список побочных (по одной в строке):\n{sub_options_text}\n\n"
        "JSON:"
    )
