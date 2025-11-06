from __future__ import annotations

"""
Назначение: регистрация хендлеров и тонкая координация переходов между шагами.
Содержит функции-обёртки, вызываемые из BotApp.start().
"""

from aiogram.filters import CommandStart, Command
from aiogram import F

from app.bot.states import Flow


def register_handlers(app) -> None:
    """Регистрирует все обработчики для текущего бота."""
    app.dp.message.register(app.cmd_start, CommandStart())
    app.dp.message.register(app.cmd_skip, Command(commands=["skip"]))
    app.dp.message.register(app.on_document, F.document)
    app.dp.callback_query.register(app.on_alt_more, F.data == "alt:more")
    app.dp.callback_query.register(app.on_actions_correct, F.data == "actions:correct")
    app.dp.callback_query.register(app.on_actions_more, F.data == "actions:more")
    app.dp.callback_query.register(app.on_actions_agree, F.data == "actions:agree")
    app.dp.callback_query.register(app.on_newclient_yes, F.data == "newclient:yes")
    app.dp.callback_query.register(app.on_newclient_no, F.data == "newclient:no")
    app.dp.callback_query.register(app.on_theme_skip, F.data == "theme:skip")
    app.dp.callback_query.register(app.on_theme_skip_more, F.data == "theme:skip_more")
    app.dp.message.register(app.on_dynamic_answer, Flow.waiting_dynamic_answer)
    app.dp.message.register(app.on_theme_answer, Flow.waiting_theme)
    app.dp.message.register(app.on_preface_step1_answer, Flow.preface1)
    app.dp.message.register(app.on_preface_step2_answer, Flow.preface2)
    app.dp.message.register(app.on_preface_step3_answer, Flow.preface3)
    app.dp.message.register(app.on_corrections_message, Flow.waiting_theme_for_corrections)


