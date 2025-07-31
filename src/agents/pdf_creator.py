import os
import requests
from PIL import Image
from io import BytesIO
from typing import Dict, Any, List, Optional
import re


# --- LangChain и LangGraph компоненты ---
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# --- Компоненты для генерации PDF ---
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.units import inch


# --- 1. Определение состояния графа (State) ---

class ReportData(TypedDict):
    """Структура для входных данных отчета."""
    graph_urls: List[str]
    graph_descriptions: List[str]
    annotations: List[str]
    news_links: List[str]
    news_articles: List[str]
    aggregated_news: List[str]


class ReportGenerationState(TypedDict):
    """
    Состояние для графа генерации отчета.
    Оно будет передаваться от узла к узлу.
    """
    messages: List[BaseMessage]      # История сообщений для LLM
    report_data: ReportData          # Исходные данные для отчета
    article_content: Optional[str]   # Сгенерированный LLM текст статьи
    pdf_path: Optional[str]          # Путь к итоговому PDF файлу



class ArticleWriterAgent:
    """
    Агент, отвечающий за написание текста статьи на основе предоставленных данных.
    Он структурирует информацию в логический рассказ.
    """
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.system_prompt = self._create_system_prompt()
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input_text}")
        ])
        self.chain = self.prompt_template | self.llm

    def _create_system_prompt(self) -> str:
        """Создает системный промпт для роли писателя-аналитика."""
        return """
            Роль: Ты высококвалифицированный эконометрист и макроэкономист. Задача: тебе надо написать большую статью на основе большой отраслевой BVAR модели. Это описание для сложной аудитории - студенты экономических специальностей и руководство центрального банка. Твои ассистенты уже описали все результаты – то есть IRF, траекторию шоков и их стандартные отклонения, историческое разложение (с разными горизонтами) включая прогноз. Результат: тебе надо создать большой документ с введением, главой про IRF, разбитой на параграфы по шокам, главой про историческое разложение, разбитой на параграфы по переменной и подпараграфы по горизонту разложения, главой по шокам и вероятно новостям им соответствующим (если твои ассистенты справятся), и самое важное – заключение на 2-3 страницы, где будет подчеркнуто самое интересное. Входные данные: тексты описания всех видов графиков и новостей подготовленные твоими ассистентами. Выход: текст статьи с рубрикацией. Дополнительная информация о модели: BVAR содержит реальные темпы роста и показатели инфляции для большого списка отраслей экономики РФ, плюс темп роста цен на нефть, процентная ставка и темп роста номинального обменного курса. Данные ежеквартальные. Структурные шоки идентифицированы с помощью знаковых ограничений, и нулевые ограничения для нефти(шоки РФ на нее не влияют, в момент их реализации). Модель включает структурный разрыв 2022 года (для всех параметров), постериор BVAR модели оцененной до разрыва после модификации используется как прайор модели после разрыва. Период оценки с 2016q2(первый год начальной выборки не используется - он был задействован для выбора числа лагов по marginalLikelihood - тут уже фактически используемые данные) по 2024q4. Все гиперпараметры BVAR модели, включая дополнительные, отвечающие за преобразование постериора до разрыва в прайор после, оценивались методом максимизации marginalLikelihood. Список отраслей следующий: A это Сельское, лесное хозяйство, охота, рыболовство и рыбоводство; B это Добыча полезных ископаемых; C это Обрабатывающие производства; F это Строительство; G это Торговля оптовая и розничная; ремонт автотранспортных средств и мотоциклов; H это Транспортировка и хранение; J это Деятельность в области информации и связи; K это Деятельность финансовая и страховая; L это Деятельность по операциям с недвижимым имуществом; M это Деятельность профессиональная, научная и техническая; O это Государственное управление и обеспечение военной безопасности; социальное обеспечение; P это Образование; Q это Деятельность в области здравоохранения и социальных услуг. А вот результаты работы твоих помошников:
            Обязательно вставляй ссылки на приложенные графики 
            Ответ предоставь в формате markdown
        """ 

    def _format_input_for_llm(self, report_data: ReportData) -> str:
        """Форматирует все входные данные в единый текстовый блок для LLM."""
        input_parts = []
        input_parts.append("### Агрегированные новости (для общего контекста):\n" + "\n".join(report_data['aggregated_news']))
        
        for i, graph_url in enumerate(report_data['graph_urls']):
            input_parts.append(f"\n--- Данные для Графика {i+1} ---")
            input_parts.append(f"Ссылка на график: {graph_url}")
            if i < len(report_data['graph_descriptions']):
                input_parts.append(f"Описание графика: {report_data['graph_descriptions'][i]}")
            if i < len(report_data['annotations']):
                input_parts.append(f"Аннотация: {report_data['annotations'][i]}")
            if i < len(report_data['news_articles']):
                input_parts.append(f"Связанная новость: {report_data['news_articles'][i]}")
                if i < len(report_data['news_links']):
                     input_parts.append(f"Источник новости: {report_data['news_links'][i]}")

        return "\n".join(input_parts)

    def __call__(self, state: ReportGenerationState) -> ReportGenerationState:
        """Вызов агента как узла графа."""
        print("--- ✍️  Вызов Агента-Писателя ---")
        
        report_data = state['report_data']
        formatted_input = self._format_input_for_llm(report_data)
        
        # Вызываем LLM для генерации статьи
        response = self.chain.invoke({"input_text": formatted_input})
        
        print("--- ✅ Статья сгенерирована ---")
        # print(response.content) # для отладки
        clear_response = response.content.replace('```markdown', '').replace('```', "")
        
        return {
            **state,
            "article_content": clear_response
        }


# --- 3. Узел для генерации PDF ---

def generate_pdf_node(state: ReportGenerationState) -> ReportGenerationState:
    """
    Узел графа, который создает PDF-файл из сгенерированной статьи и графиков.
    """
    print("--- 📄 Вызов Генератора PDF ---")
    
    article_text = state.get("article_content")
    if not article_text:
        raise ValueError("Текст статьи не был сгенерирован.")
        
    report_data = state['report_data']
    output_filename = "analytical_report.pdf"
    
    # Настройки документа
    c = canvas.Canvas(output_filename, pagesize=letter)
    width, height = letter
    styles = getSampleStyleSheet()
    
    # Добавим кастомный стиль для заголовка
    styles.add(ParagraphStyle(name='ReportTitle', fontSize=24, leading=28, alignment=1, spaceAfter=20))
    
    story = []
    
    # Заголовок
    story.append(Paragraph("Аналитический отчет", styles['ReportTitle']))

    # Разделяем текст по плейсхолдерам графиков
    text_parts = article_text.split('[GRAPH_')
    
    # Добавляем первую часть текста
    story.append(Paragraph(text_parts[0].replace('\n', '<br/>'), styles['BodyText']))
    story.append(Spacer(1, 0.2 * inch))

    # Обрабатываем остальные части и вставляем графики
    for i, part in enumerate(text_parts[1:]):
        # part выглядит как "1] Текст после первого графика..."
        # Нам нужно извлечь текст после закрывающей скобки
        try:
            text_after_graph = part.split(']', 1)[1]
            graph_index = i
            
            # Скачиваем и вставляем изображение
            if graph_index < len(report_data['graph_urls']):
                url = report_data['graph_urls'][graph_index]
                try:
                    print(f"--- 📥 Скачивание графика: {url} ---")
                    response = requests.get(url)
                    response.raise_for_status() # Проверка на ошибки HTTP
                    
                    # Используем Pillow для определения размеров и сохранения
                    pil_image = Image.open(BytesIO(response.content))
                    
                    # Масштабируем изображение, чтобы оно поместилось на страницу
                    max_width = width - 2 * inch
                    max_height = height / 3 # Ограничим высоту, чтобы не занимать всю страницу
                    pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

                    # Сохраняем во временный файл для ReportLab
                    img_path = f"temp_graph_{graph_index}.png"
                    pil_image.save(img_path)
                    
                    story.append(ReportLabImage(img_path, width=pil_image.width, height=pil_image.height))
                    story.append(Spacer(1, 0.2 * inch))

                except Exception as e:
                    print(f"--- ⚠️ Не удалось загрузить или обработать график {url}: {e} ---")
                    story.append(Paragraph(f"[Ошибка: не удалось загрузить график {graph_index + 1}]", styles['BodyText']))

            # Добавляем текст после графика
            story.append(Paragraph(text_after_graph.replace('\n', '<br/>'), styles['BodyText']))
            story.append(Spacer(1, 0.2 * inch))

        except IndexError:
            # Если в части нет ']', просто добавляем ее как текст
            story.append(Paragraph(part.replace('\n', '<br/>'), styles['BodyText']))
    
    # Собираем документ
    story_frame = c.beginText(inch, height - inch)
    doc_width = width - 2*inch
    
    # Это упрощенная отрисовка. Для сложных документов лучше использовать platypus.Frame
    y_pos = height - inch
    for item in story:
        item_height = item.wrap(doc_width, height)[1]
        if y_pos - item_height < inch:
            c.showPage() # Новая страница
            y_pos = height - inch
        
        item.drawOn(c, inch, y_pos - item_height)
        y_pos -= (item_height + item.style.spaceAfter)

    c.save()
    
    # Удаляем временные файлы изображений
    for i in range(len(report_data['graph_urls'])):
        img_path = f"temp_graph_{i}.png"
        if os.path.exists(img_path):
            os.remove(img_path)
            
    print(f"--- ✅ PDF файл сохранен как: {output_filename} ---")
    
    return {
        **state,
        "pdf_path": output_filename
    }

