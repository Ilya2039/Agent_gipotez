from __future__ import annotations


def build_decide_next_action_prompt(
    theme: str,
    qa_json: str,
    dialog_json: str = "",
    prefer_non_finance: bool = True,
) -> str:
    context_block = f"\nКонтекст (JSON из файла, укорочен):\n{dialog_json}\n" if dialog_json else ""
    deprior = (
        "Финансовые гипотезы считаются наименее приоритетными.\n"
        if prefer_non_finance
        else ""
    )
    return (
        "РОЛЬ: Старший консультант по стратегии развития бизнеса с отраслевой специализацией.\n"
        + "ЦЕЛЬ: вести диагностический диалог, минимизируя шум и повторения, и принимать решение о следующем шаге.\n"
        + "ФОКУС: отраслевые тренды, специфика подотрасли, практики конкурентов, операционные и организационные особенности ведения бизнеса.\n"
        + deprior
        + "ЗАДАЧА: по истории Q/A выбрать действие — задать вопрос, запросить материалы, сформулировать гипотезу или завершить.\n"
        + "СТРОГИЙ ВЫВОД (только JSON):\n"
        + '{"action":"ask|request_file|hypothesis|stop","question":"<если action=ask>","note":"<кратко почему>"}'
        + "\nКРИТЕРИИ:\n"
        + "- Недостаточно данных → action=ask и 1 прикладной вопрос.\n"
        + "- Вопрос ≤ 18 слов, по-деловому, учитывает контекст, обязателен знак '?'.\n"
        + "- Явный дефицит данных/цифр → action=request_file.\n"
        + "- Устойчивая картинка и отраслевой смысл → action=hypothesis.\n"
        + f"Тема: {theme}\n"
        + f"История (JSON массив объектов {{question, answer}}):\n{qa_json}\n"
        + f"{context_block}"
        + "JSON:"
    )


def build_generate_discovery_question_prompt(
    qa_json: str,
    dialog_json: str = "",
    examples_text: str = "",
    avoid: list[str] | None = None,
    count: int = 1,
) -> str:
    context_block = f"\nКонтекст (JSON из файла, укорочен):\n{dialog_json}\n" if dialog_json else ""
    examples_block = (
        "\nПримеры типовых гипотез (для ориентира; НЕ выбирай их, формулируй свои вопросы):\n"
        + examples_text
        + "\n"
        if examples_text
        else ""
    )
    avoid_block = (
        ("\nНе повторяй и не перефразируй эти вопросы:\n- " + "\n- ".join(avoid)) if avoid else ""
    )
    return (
        "РОЛЬ: Старший интервьюер/консалтер.\n"
        + "ЦЕЛЬ: уточнить деятельность компании: отраслевая специфика, продукты, конкуренты, клиенты, процессы, задачи, сложности и боли.\n"
        + "ТРЕБОВАНИЯ К ВОПРОСУ: без общих слов, ≤ 18 слов, оканчивается на '?'.\n"
        + "Учитывай предыдущие ответы, не дублируй. Спрашивай о разных сферах ведения бизнеса, расширяй фокус.\n"
        + "СТРОГИЙ ВЫВОД — только JSON‑массив из ровно N вопросов без текста вокруг.\n"
        + f"N={count}.\n"
        + f"История Q/A (JSON):\n{qa_json}\n"
        + f"{context_block}"
        + f"{examples_block}"
        + f"{avoid_block}"
        + "Формат ответа: [\"Вопрос?\"]"
    )

def build_generate_free_hypothesis_prompt(
    theme: str,
    qa_json: str,
    dialog_json: str = "",
    examples_text: str = "",
    prefer_non_finance: bool = True,
) -> str:
    context_block = f"\nКонтекст (JSON из файла, укорочен):\n{dialog_json}\n" if dialog_json else ""
    examples_block = (
        "\nПримеры типовых гипотез (для ориентира; можно формулировать свои):\n"
        + examples_text
        + "\n"
        if examples_text
        else ""
    )
    deprior = (
        "\nФинансовые гипотезы — низкий приоритет. Не предлагай их без прямых фактов в их пользу.\n"
        if prefer_non_finance
        else ""
    )
    return (
        "РОЛЬ: Ведущий отраслевой стратег.\n"
        + "ЦЕЛЬ: сформулировать одну прикладную гипотезу по теме с опорой на отраслевую специфику и данные о конкретном клиенте.\n"
        + "ТРЕБОВАНИЯ: гипотеза конкретная, проверяемая, ведёт к действиям; избегать общих финансовых ярлыков. Гипотеза - не ответ на вопросы, а следствие.\n"
        + deprior
        + "СТРОГИЙ ВЫВОД — только JSON: {\n"
        + '  "hypothesis": "<краткая формулировка>",\n'
        + '  "reason": "<10-20 слов, на какие факты опираешься>",\n'
        + '  "tags": ["отрасль", "подотрасль", "процессы"]\n'
        + "}\n"
        + f"Тема: {theme}\n"
        + f"История (Q/A JSON):\n{qa_json}\n"
        + f"{context_block}"
        + f"{examples_block}"
        + "JSON:"
    )


def build_generate_meeting_questions_prompt(
    hypothesis: str,
    qa_json: str,
    dialog_json: str = "",
    count: int = 4,
    avoid_questions: list[str] | None = None,
) -> str:
    context_block = f"\nКонтекст (JSON из файла, укорочен):\n{dialog_json}\n" if dialog_json else ""
    avoid_block = (
        ("\nНе повторяй и не перефразируй эти вопросы:\n- " + "\n- ".join(avoid_questions)) if avoid_questions else ""
    )
    return (
        "РОЛЬ: Партнёр на встрече с клиентом.\n"
        + "ЦЕЛЬ: подготовить 3–4 проверочных вопроса для валидации гипотезы и уточнения предпосылок.\n"
        + "ТРЕБОВАНИЯ: вопросы прикладные, отраслево-уместные, без общих фраз и дисклеймеров; ≤ 18 слов; знак '?'.\n"
        + "СТРОГИЙ ВЫВОД — только JSON‑массив из ровно N строк без текста вокруг.\n"
        + f"N={count}.\n"
        + f"Гипотеза: {hypothesis}\n"
        + f"История Q/A (JSON):\n{qa_json}\n"
        + f"{context_block}"
        + f"{avoid_block}"
        + "Формат ответа: [\"Вопрос?\", ...]"
    )


def build_generate_alternative_hypothesis_prompt(
    qa_json: str,
    dialog_json: str = "",
    examples_text: str = "",
    avoid_hypothesis: str = "",
    prefer_non_finance: bool = True,
) -> str:
    context_block = f"\nКонтекст (JSON из файла, укорочен):\n{dialog_json}\n" if dialog_json else ""
    examples_block = (
        "\nПримеры типовых гипотез (для ориентира; можно формулировать свои):\n"
        + examples_text
        + "\n"
        if examples_text
        else ""
    )
    deprior = (
        "\nФинансовые гипотезы — низкий приоритет. Не предлагай их без прямых фактов в их пользу.\n"
        if prefer_non_finance
        else ""
    )
    avoid_block = f"\nИсключить гипотезу: {avoid_hypothesis}\n" if avoid_hypothesis else ""
    return (
        "РОЛЬ: Ведущий отраслевой стратег.\n"
        + "ЦЕЛЬ: предложить альтернативную прикладную гипотезу, отличную от указанной, с опорой на отраслевую специфику.\n"
        + "ТРЕБОВАНИЯ: гипотеза конкретная, проверяемая, ведёт к действиям; избегать общих финансовых ярлыков.\n"
        + deprior
        + "СТРОГИЙ ВЫВОД — только JSON: {\n"
        + '  "hypothesis": "<краткая формулировка>",\n'
        + '  "reason": "<10-20 слов, на какие факты опираешься>",\n'
        + '  "tags": ["отрасль", "подотрасль", "процессы"]\n'
        + "}\n"
        + f"История (Q/A JSON):\n{qa_json}\n"
        + f"{context_block}"
        + f"{examples_block}"
        + f"{avoid_block}"
        + "JSON:"
    )


