from docx import Document
from typing import List
from app.core.logger import log


def parse_docx(file_path: str) -> List[str]:
    """
    Парсит docx файл и возвращает список параграфов текста.
    """
    try:
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        log.info(f"Файл '{file_path}' успешно прочитан, {len(paragraphs)} параграфов найдено.")
        return paragraphs
    except Exception as e:
        log.error(f"Ошибка при чтении файла '{file_path}': {e}")
        return []
