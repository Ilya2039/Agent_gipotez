
import os
import yaml
import json
import app.core.utils as utils
from typing import Optional, List
from app.core.llm_client import GigaChatClient
from app.core.logger import log

with open("cfg/prompts.yaml", "r") as f:
    prompts_cfg = yaml.safe_load(f)

with open("cfg/common.yaml", "r") as f:
    common_cfg = yaml.safe_load(f)


class ClientPipeline:
    def __init__(self):
        log.info("Инициализация пайплайна ClientPipeline")
        self.gigachat_client = GigaChatClient()
        self.docx_content = ""
        self.arrangements = {}
        self.additional_context = ''
        self.theme = ''
        self.hypothesis = ''
        self.questions = []

    def process_docx(self, folder_path: str) -> str:
        log.info(f"Начало обработки docx файлов: {folder_path}")
        docx_files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.endswith(".docx")
            ]
        log.info(f"Найдено {len(docx_files)} docx файлов для обработки")
        all_docx = []
        for file_path in docx_files:
            paragraphs = utils.parse_docx(file_path)
            combined_text = "\n".join(paragraphs)
            all_docx.append(combined_text)
            log.info(f"Файл '{file_path}' обработан")
        self.docx_content = "\n".join(all_docx)
        log.info("Обработка docx файлов завершена")
        return self.docx_content

    def find_arrangements(self, query: str = 'all') -> str:
        log.info("Начало поиска договоренностей в тексте")
        prompt = f"""
            {prompts_cfg['user_prompts']['arrangements'][query]}
            {self.docx_content}
        """
        response = self.gigachat_client.invoke(prompt)
        log.info("Gigachat дал ответ")
        try:
            response = json.loads(response)
            answer = '\n'.join([f"{i + 1}. {x}" for i, x in enumerate(response)])
        except json.JSONDecodeError:
            answer = response
        log.info("Поиск договоренностей завершен")
        self.arrangements[query] = answer
        return answer

    def generate_questions(self, theme: Optional[str] = None) -> str:
        log.info("Формирование вопросов на основе имеющейся информации")
        if theme is not None:
            self.theme = f"Особенно сфокусируйся на теме: {theme}."
        
        prompt_template = prompts_cfg["user_prompts"]["hypothesis"]["ask_questions"]

        prompt = f"""{prompt_template}
docx_content: {self.docx_content}
additional_context: {self.additional_context}
theme: {self.theme}
"""
        response = self.gigachat_client.invoke(prompt)
        try:
            self.questions = json.loads(response)
        except json.JSONDecodeError:
            log.error("Ошибка при разборе JSON с вопросами")
            log.info(f'Ответ GigaChat: {response}')
            self.questions = []
        log.info("Формирование вопросов завершено")
        return response

    def make_hypothesis(self) -> List[str]:
        log.info("Начало формирования гипотез")
        prompt_template = prompts_cfg["user_prompts"]["hypothesis"]["prompt"]

        prompt = f"""{prompt_template}
docx_content: {self.docx_content}
additional_context: {self.additional_context}
theme: {self.theme}
"""
        response = self.gigachat_client.invoke(prompt)
        log.info("Формирование гипотез завершено")
        log.info(f'Ответ GigaChat: {response}')
        
        try:
            hypotheses = json.loads(response)
            if isinstance(hypotheses, list):
                self.hypothesis = hypotheses
                return hypotheses
            elif isinstance(hypotheses, dict) and "hypotheses" in hypotheses:
                self.hypothesis = hypotheses["hypotheses"]
                return hypotheses["hypotheses"]
            else:
                return [response]
        except json.JSONDecodeError:
            log.error("Ошибка при разборе JSON с гипотезами")
            return [response]
