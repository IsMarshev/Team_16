from __future__ import annotations

# --- Стандартная библиотека ---
import os
import uuid
import requests
from io import BytesIO
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from dotenv import load_dotenv
import numpy as np

# --- Работа с данными и графиками ---
import pandas as pd
from PIL import Image

# --- Парсинг HTML и Markdown ---
import markdown
from bs4 import BeautifulSoup

# --- LangChain и LangGraph компоненты ---
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch

# --- Компоненты агентов ---
from src.agents import (
    AgentState,
    RoleAgent,
    SearchAgentState,
    SearchAgent,
    ArticleWriterAgent,
    ReportGenerationState,
    ReportData,
)
from src.config import Config

# --- Генерация PDF ---
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

load_dotenv()


class PromptState(TypedDict):
    """Состояние для графа"""
    messages: List[BaseMessage]

def _merge_dicts(left: dict, right: dict) -> dict:
    return {**left, **right}

class RootState(TypedDict):
    describer_state: Annotated[AgentState, _merge_dicts]
    summarizer_state: Annotated[AgentState, _merge_dicts] 
    prompt_state: Annotated[PromptState, _merge_dicts]
    search_state: Annotated[SearchAgentState, _merge_dicts]
    pdf_state: Annotated[ReportGenerationState, _merge_dicts]


config = Config()

llm = ChatOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.url,
            model="gemini-2.5-pro-preview",
            temperature=config.websearch.temperature,
            model_kwargs={"max_tokens": 32000},
            extra_body={
                "use_beam_search": config.websearch.use_beam_search, 
                "best_of": config.websearch.best_of
            }
        )

summurize_llm = ChatOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.url,
            model="qwen3-235b-a22b",
            temperature=config.websearch.temperature,
            model_kwargs={"max_tokens": 32000},
            extra_body={
                "use_beam_search": config.websearch.use_beam_search, 
                "best_of": config.websearch.best_of
            }
        )

article_llm = ChatOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.url,
            model="gpt-4.1-mini",
            temperature=config.websearch.temperature,
            model_kwargs={"max_tokens": 32000},
            extra_body={
                "use_beam_search": config.websearch.use_beam_search, 
                "best_of": config.websearch.best_of
            }
        )


describer_agent = RoleAgent(
    llm=llm,
    role_name="Ассистент",
    role_description="You are high skill econometrist and macroeconomist",
    role_instructions=[
        "You are high skill econometrist and macroeconomist. You need describe IRF from large BVAR. It is for economic students and staff of central bank. BVAR model includes real growth rates and inflation of many russian industries plus oil prices growth rate, interest rate and nominal exchange growth rate. Structural shocks are identified with sign restriction plus zero restriction for oil prices. The model include structural break (for all parameters). The list of industries is following: A is  Agriculture, forestry, fishing, and hunting; B is  Mining; C is industrial production; F is Construction; G is  Retail trade, Wholesale trade; H is  Transportation and warehousing; J is  Information; K is  Finance and insurance; L is  Real estate and rental and leasing; M is  Professional and business services, science; O is Government; P is Education; Q is Health care, and social assistance; There are a lot of plots. This plot(includes about 9 subplots with 2 lines on each) that required comments includes IRF for following shock: dpoil. You need describe IRFs and give economic intuition (why such shock should have such influence on the variable). Highlight cross industry effects. Subplot includes response of following variable: dfx. Before break response is following:"
    ],
    additional_context="Ты встроен в граф задач."
)

summarizer_agent = RoleAgent(
    llm=summurize_llm,
    role_name="Ассистент",
    role_description="Ты профессиональный рерайтер",
    role_instructions=[
        "Тебе надо суммаризовать описание в краткую аннотацию",
        "Текст надо сокращать с умом, чтобы не потерять важную информацию",
    ],
    additional_context="Ты встроен в граф задач."
)

new_agregate_agent = RoleAgent(
    llm=summurize_llm,
    role_name="Ассистент",
    role_description="Ты профессиональный рерайтер",
    role_instructions=[
        "Тебе надо суммаризовать описание в краткую аннотацию",
        "Текст надо сокращать с умом, чтобы не потерять важную информацию",
    ],
    additional_context="Ты встроен в граф задач."
)

prompt_writer = RoleAgent(
    llm=llm,
    role_name="PromptWriter",
    role_description="Генерирует поисковый запрос по заданной теме",
    role_instructions=[
        "Прочитай сообщение пользователя и придумай, что именно стоит искать в интернете.",
        "Верни чистый поисковый запрос без лишних слов."
    ]
)

search_agent = SearchAgent(config=config)

article_agent = ArticleWriterAgent(llm=article_llm)

class Configuration(TypedDict):
    """Configurable parameters for the agent."""
    # API key for the TavilySearch service
    tavily_api_key: str
    # OpenAI API key for ChatOpenAI
    openai_api_key: str
    # Model name for ChatOpenAI (e.g., "gpt-3.5-turbo")
    model_name: str

def parse_values(s):
    values = []
    for item in s.split(','):
        item = item.strip()
        if item == 'NaN':
            values.append(np.nan)
        else:
            try:
                values.append(float(item))
            except:
                values.append(item)
    return values

def process_excel_data(file_path):

    df = pd.read_excel(file_path, header=None, engine='openpyxl')

    result = {}
    for idx, row in df.iterrows():
        group_name = row[1]
        content = str(row[0])

        if pd.isna(content) or content.strip() == "":
            continue

        shock_blocks = content.split('Subplot includes the following shock:')[1:]
        group_data = []

        for block in shock_blocks:
            shock_line = block.strip().split('\n')[0]
            shock_name = shock_line.split('.')[0].strip()

            if 'Mean is following:' not in block or 'Std is following:' not in block:
                continue

            mean_part = block.split('Mean is following:')[1].split('Std is following:')[0].strip()
            std_part = block.split('Std is following:')[1].split('Subplot includes the following shock:')[0].strip()

            mean_values = parse_values(mean_part)
            std_values = parse_values(std_part)


            shock_data = {
                "data": {
                    "mean": mean_values,
                    "std": std_values
                }
            }
            group_data.append(shock_data)

        result[group_name] = group_data

    return result


def run_load_and_process(root: RootState) -> RootState:
    # 1. Читаем Excel (путь можно передавать через root.context или брать статично)
    df = pd.read_excel(root["describer_state"]["context"]["excel_path"],
                       usecols=["prompt", "fig_name"])
    
    records: list[dict] = []
    for _, row in df.iterrows():
        # 2. Для каждой строки готовим AgentState
        state: AgentState = {
            "messages": [HumanMessage(content=row["prompt"])],
            "context": {}
        }
        # 3. Запускаем describer_agent
        result = describer_agent(state)
        description = result["messages"][-1].content
        graphic_url = result.get("context", {}).get("graphic_url", "")
        
        # 4. Собираем запись
        records.append({
            "id":           row["fig_name"],
            "describe":     description,
            "graphic_url":  f"/Users/ilya/Desktop/work/mlforum/plots/{row["fig_name"]}.png"
        })


    df_out = pd.DataFrame(records)
    out_path = "temp/all_descriptions.parquet"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_parquet(out_path, index=False, engine="fastparquet")
    print(f"Saved {len(df_out)} records to {out_path}")

    return root



def run_summarize(root: RootState) -> RootState:
    df = pd.read_parquet('temp/all_descriptions.parquet', engine='fastparquet')
    
    summarizes = []
    for item in range(len(df)):
        state: AgentState = {
            "messages": [HumanMessage(content=df['describe'][item])],
            "context": {}
        }
        result = summarizer_agent(state)
        summarizes.append(result["messages"][-1].content)
    df['summarize'] = summarizes
    out_path = 'temp/all_descriptions_and_summary.parquet'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False, engine="fastparquet")
    print(f"Saved {len(df)} records to {out_path}")
    return root



def write_prompt(root: RootState) -> RootState:
    new_state = prompt_writer(root["prompt_state"])
    root["prompt_state"] = new_state
    # допустим, prompt_writer кладёт сформулированный поисковый запрос 
    # в последний message.content, переносим его в контекст для поискового агента:
    last = new_state["messages"][-1].content
    # инициализируем вход для SearchAgent
    root["search_state"] = {
        "search_query": last,
        "search_results": [] 
    }
    print(root['search_state'])
    return root

def run_search(root: RootState) -> RootState:
    new_search = search_agent(root["search_state"])
    root["search_state"] = new_search

    search_results = new_search["search_results"]
    records = []
    for result in search_results:
        if isinstance(result, dict):
            records.append({
                "title": result.get("title", ""),
                "texts": result.get("content", ""),
                "url": result.get("url", ""),
                # "date": result.get("date", ""),            
                # "sector": result.get("sector", ""),          
                "relevant score": result.get("relevance_score", 0.0),
            })
        else:
            records.append({
                "title": result.title,
                "texts": result.content,
                "url": result.url,
                # "date": getattr(result, "date", ""),      
                # "sector": getattr(result, "sector", ""),  
                "relevant score": result.relevance_score,
            })

    df = pd.DataFrame(records)

    out_path = "temp/search_results.parquet"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False, engine="fastparquet")
    print(f"✅ Saved {len(df)} search results to {out_path}")
    
    return root

def run_news_summarize(root: RootState) -> RootState:
    # Загружаем результаты поиска
    df = pd.read_parquet('temp/search_results.parquet', engine='fastparquet')

    # Объединяем все тексты в один большой блок для единого обобщённого резюме
    combined_items = []
    for idx, row in df.iterrows():
        combined_items.append({
            "title": row['title'],
            "texts": row['texts'],
            "url": row['url']
        })

    # Формируем один запрос для агрегированного резюме
    payload = combined_items
    state: AgentState = {
        "messages": [HumanMessage(content=str(payload))],
        "context": {}
    }

    # Получаем единый, крупный обобщённый резюме по всем статьям
    result = new_agregate_agent(state)
    big_summary = result["messages"][-1].content

    # Сохраняем вместе с исходными данными
    df['summarize'] = big_summary
    out_path = 'temp/search_results_and_aggregate_summary.parquet'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False, engine="fastparquet")

    print(f"Saved aggregate summary for {len(df)} records to {out_path}")
    return root


def run_article_agent(root: RootState) -> RootState:
    df_graphs = pd.read_parquet("/Users/ilya/Desktop/work/mlforum/Team_16/temp/all_descriptions_and_summary.parquet")
    df_summaries = pd.read_parquet("/Users/ilya/Desktop/work/mlforum/Team_16/temp/search_results_and_aggregate_summary.parquet")

    # Формируем структуру отчета
    report: ReportData = {
        'graph_urls': df_graphs['graphic_url'].tolist(),
        'graph_descriptions': df_graphs['describe'].tolist(),
        'annotations': df_graphs['summarize'].tolist(),
        'news_links': df_summaries['url'].tolist(),
        'news_articles': df_summaries['texts'].tolist(),
        'aggregated_news': df_summaries['summarize'].tolist(),
    }

    # Сохраняем в root
    root['pdf_state']['report_data'] = report
    result = article_agent(root["pdf_state"])
    root["pdf_state"] = result
    return root

def setup_fonts():
    """
    Скачивает (если нужно) и регистрирует семейство шрифтов DejaVu
    для поддержки кириллицы и начертаний (жирный, курсив).
    """
    print("--- ⚙️  Настройка шрифтов для PDF ---")
    
    # Определяем все необходимые шрифты и их URL для скачивания
    font_variants = {
        'DejaVuSans': 'https://github.com/mps/fonts/blob/master/DejaVuSans.ttf?raw=true',
        'DejaVuSans-Bold': 'https://github.com/mps/fonts/blob/master/DejaVuSans-Bold.ttf?raw=true',
        'DejaVuSans-Oblique': 'https://github.com/mps/fonts/blob/master/DejaVuSans-Oblique.ttf?raw=true',
        'DejaVuSans-BoldOblique': 'https://github.com/mps/fonts/blob/master/DejaVuSans-BoldOblique.ttf?raw=true',
    }

    for font_name, url in font_variants.items():
        font_path = f"fonts/{font_name}.ttf"
        if not os.path.exists(font_path):
            print(f"--- 📥 Шрифт {font_path} не найден, скачиваю...")
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(font_path, 'wb') as f:
                    f.write(response.content)
                print(f"--- ✅ Шрифт {font_path} успешно скачан.")
            except Exception as e:
                raise IOError(f"Не удалось скачать шрифт {url}: {e}")
        
        # Регистрируем каждый шрифт
        pdfmetrics.registerFont(TTFont(font_name, font_path))

    # Связываем шрифты в одно семейство. Это ключ к работе <b> и <i>
    pdfmetrics.registerFontFamily(
        'DejaVuSans',
        normal='DejaVuSans',
        bold='DejaVuSans-Bold',
        italic='DejaVuSans-Oblique',
        boldItalic='DejaVuSans-BoldOblique'
    )

def html_to_flowables(html_text, styles):
    """
    Исправленная версия, которая корректно итерируется по тегам внутри <body>
    и сохраняет вложенную разметку.
    """
    # 1. ReportLab понимает <b> и <i>, поэтому эта замена полезна.
    html_text = (
        html_text
        .replace('<strong>', '<b>').replace('</strong>', '</b>')
        .replace('<em>', '<i>').replace('</em>', '</i>')
    )
    
    soup = BeautifulSoup(html_text, 'lxml')
    flowables = []

    # 2. Находим тело документа. Если его нет, работаем с тем что есть.
    body = soup.find('body')
    if not body:
        # Если вдруг пришел фрагмент HTML без <body>, используем сам soup
        body = soup

    # 3. Итерируемся по ПРЯМЫМ потомкам <body>. Это правильная логика.
    for tag in body.find_all(True, recursive=False):
        tag_name = tag.name.lower()
        print(tag_name)
        if tag_name in ['h1', 'h2', 'h3', 'h4']:
            style_name = tag_name.upper() # H1, H2 и т.д.
            if style_name in styles:
                flowables.append(Spacer(1, 0.15 * inch))
                flowables.append(Paragraph(tag.decode_contents(), styles[style_name]))
                flowables.append(Spacer(1, 0.1 * inch)) # Небольшой отступ после заголовка
            else:
                 flowables.append(Paragraph(tag.decode_contents(), styles['BodyTextCustom']))

        elif tag_name == 'p':
            flowables.append(Paragraph(tag.decode_contents(), styles['BodyTextCustom']))

        elif tag_name in ['ul', 'ol']:
            # Ваша логика для списков хороша для простых случаев
            list_items = []
            for li in tag.find_all('li'):
                # Создаем один Paragraph для всего списка для лучшего форматирования
                # Используем <br/> для переносов строк внутри одного параграфа
                list_items.append(f"• {li.decode_contents()}")
            
            # Собираем все элементы списка в один параграф
            if list_items:
                full_list_text = "<br/>".join(list_items)
                flowables.append(Paragraph(full_list_text, styles['ListItem']))
        
        # Если есть просто текст без тегов между блоками, его нужно тоже обработать
        elif tag.name is None and tag.string.strip():
             flowables.append(Paragraph(tag.string, styles['BodyTextCustom']))

        else:
            # Для неизвестных тегов (div, etc) просто рендерим их содержимое
            flowables.append(Paragraph(tag.decode_contents(), styles['BodyTextCustom']))
            
    return flowables

def generate_pdf_node(root: RootState) -> RootState:
    """
    Узел графа, который создает PDF-файл из сгенерированной статьи,
    поддерживая кириллицу и полную Markdown разметку (включая заголовки).
    """
    # --- 1. Настраиваем шрифты (предполагаем, что функция setup_fonts существует) ---
    setup_fonts()
    
    print("--- 📄 Вызов Генератора PDF с полной поддержкой Markdown ---")

    article_text = root['pdf_state'].get("article_content")
    if not article_text:
        raise ValueError("Текст статьи не был сгенерирован.")
        
    report_data = root['pdf_state']['report_data']
    output_filename = "analytical_report.pdf"
    
    # --- 2. Определяем ВСЕ необходимые стили ---
    FONT_FAMILY_NAME = 'DejaVuSans'
    styles = getSampleStyleSheet()

    # Стиль заголовка уровня 1 — жирным
    styles.add(ParagraphStyle(
        name='H1',
        parent=styles['h1'],
        fontName='DejaVuSans-Bold',  # Жирный
        fontSize=20,
        leading=24,
        spaceAfter=16
    ))
    styles.add(ParagraphStyle(
        name='H2',
        parent=styles['h2'],
        fontName='DejaVuSans-Bold',
        fontSize=18,
        leading=22,
        spaceAfter=14
    ))
    styles.add(ParagraphStyle(
        name='H3',
        parent=styles['h3'],
        fontName='DejaVuSans-Bold',
        fontSize=16,
        leading=20,
        spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        name='H4',
        parent=styles['h4'],
        fontName='DejaVuSans-Bold',
        fontSize=14,
        leading=18,
        spaceAfter=10
    ))
    
    # Основные стили
    styles.add(ParagraphStyle(name='ReportTitle', fontName=FONT_FAMILY_NAME, fontSize=24, leading=28, alignment=1, spaceAfter=20))
    styles.add(ParagraphStyle(name='BodyTextCustom', parent=styles['BodyText'], fontName=FONT_FAMILY_NAME, spaceAfter=12))
    
    
    # Стиль для элементов списка
    styles.add(ParagraphStyle(name='ListItem', parent=styles['BodyText'], fontName=FONT_FAMILY_NAME, leftIndent=20, spaceAfter=6))


    # --- 3. Собираем документ (story) ---
    story = []
    story.append(Paragraph("Аналитический отчет", styles['ReportTitle']))

    text_parts = article_text.split('[GRAPH_')
    
    # Конвертируем первую часть текста
    html_part = markdown.markdown(text_parts[0])
    story.extend(html_to_flowables(html_part, styles))
    
    # Обрабатываем остальные части с графиками
    for i, part in enumerate(text_parts[1:]):
        try:
            text_after_graph = part.split(']', 1)[1]
            graph_index = i
            
            if graph_index < len(report_data['graph_urls']):
                url = report_data['graph_urls'][graph_index]
                try:
                    # Вставляем график
                    print(f"--- 📥 Скачивание графика: {url} ---")
                    with open(url, "rb") as f:
                        img_data = f.read()
                    pil_image = Image.open(BytesIO(img_data))
                    width, height = letter
                    max_width = width - 2 * inch
                    pil_image.thumbnail((max_width, max_width), Image.Resampling.LANCZOS)
                    story.append(Spacer(1, 0.2 * inch))
                    story.append(ReportLabImage(BytesIO(img_data), width=pil_image.width, height=pil_image.height))
                    story.append(Spacer(1, 0.2 * inch))
                except Exception as e:
                    print(f"--- ⚠️ Не удалось загрузить или обработать график {url}: {e} ---")
                    error_html = markdown.markdown(f"**[Ошибка:** не удалось загрузить график {graph_index + 1}]")
                    story.extend(html_to_flowables(error_html, styles))
            
            # Конвертируем текст ПОСЛЕ графика
            html_after_graph = markdown.markdown(text_after_graph)
            story.extend(html_to_flowables(html_after_graph, styles))

        except IndexError:
            html_part_fallback = markdown.markdown(part)
            story.extend(html_to_flowables(html_part_fallback, styles))

    # --- 4. Сборка PDF ---
    doc = SimpleDocTemplate(
        output_filename, pagesize=letter, rightMargin=inch,
        leftMargin=inch, topMargin=inch, bottomMargin=inch
    )
    doc.build(story)
    
    print(f"--- ✅ PDF файл с полной поддержкой Markdown сохранен как: {output_filename} ---")
    
    # Возвращаем обновленное состояние
    updated_pdf_state = root['pdf_state'].copy()
    updated_pdf_state["pdf_path"] = output_filename
    new_root = root.copy()
    new_root['pdf_state'] = updated_pdf_state
    return new_root

def barrier_router(state: RootState) -> str:
    return state

def clear_temp(state: RootState) -> str:
    import shutil

    path ="/Users/ilya/Desktop/work/mlforum/Team_16/temp"
    if os.path.exists(path) and os.path.isdir(path):
        try:
            shutil.rmtree(path)
            print(f"Папка '{path}' успешно удалена.")
        except Exception as e:
            print(f"Ошибка при удалении папки '{path}': {e}")
    else:
        print(f"Папка '{path}' не найдена.")


graph = (
    StateGraph(RootState, config_schema=Configuration)
      .add_node("describer_node", run_load_and_process)
      .add_node("summarize_describer_node", run_summarize)
      .add_node("prompt_node", write_prompt)
      .add_node("search_node", run_search)
      .add_node("news_aggregate", run_news_summarize)
      .add_node("generate_pdf_node", generate_pdf_node)
      .add_node("run_article_agent", run_article_agent)
      .add_node("barrier_node", barrier_router)
      .add_node("clear_temp", clear_temp)

      .add_edge("__start__", "describer_node")
      .add_edge("__start__", "prompt_node")

      .add_edge("describer_node", "summarize_describer_node")
      .add_edge("summarize_describer_node", "barrier_node") 

      .add_edge("prompt_node", "search_node")
      .add_edge("search_node", "news_aggregate")

      .add_edge("news_aggregate", "run_article_agent")
      .add_edge("barrier_node", "run_article_agent")

      .add_edge("run_article_agent", "generate_pdf_node")
      .add_edge("generate_pdf_node",  "clear_temp")
      .add_edge("clear_temp",  END)
      .compile(name="Parallel Two‑State Graph with Aggregate Summary")
)


