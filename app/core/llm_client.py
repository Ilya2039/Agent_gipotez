import yaml
import os
from dotenv import load_dotenv
from langchain_community.chat_models.gigachat import GigaChat as LCGigaChat
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.logger import log
from typing import Optional


with open("cfg/common.yaml", "r") as f:
    common_cfg = yaml.safe_load(f)

load_dotenv()


class GigaChatClient:
    def __init__(self):
        log.info("Инициализация GigaChat клиента с общими настройками из cfg/common.yaml")
        auth = os.getenv("GIGACHAT_AUTH")
        if not auth:
            log.error("GIGACHAT_AUTH не задан")
            raise RuntimeError("GIGACHAT_AUTH не найден в окружении")

        self.client = LCGigaChat(
            credentials=auth,
            **common_cfg["llm"]
        )
        log.info("GigaChat клиент успешно инициализирован")

    def invoke(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        log.info("Отправка запроса в GigaChat")
        messages = []
        if system_prompt is not None:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_prompt))
        response = self.client.invoke(messages)
        try:
            result = response.content  # type: ignore[attr-defined]
            log.info("Получен ответ от GigaChat")
            return result
        except Exception as e:
            log.error(f"Ошибка при вызове GigaChat: {e}")
            raise
