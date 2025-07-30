from __future__ import annotations
from langchain_core.messages import BaseMessage, HumanMessage
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, TypedDict
from langgraph.graph import END
import pandas as pd
import uuid


from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from typing import Dict, Any, List, Optional
from src.agents import AgentState, RoleAgent
from src.agents import SearchAgentState, SearchAgent
from src.config import Config
import os
from dotenv import load_dotenv

load_dotenv()


class PromptState(TypedDict):
    """Состояние для графа"""
    messages: List[BaseMessage]

def _merge_dicts(left: dict, right: dict) -> dict:
    return {**left, **right}

class RootState(TypedDict):
    # For each key, we wrap its type in Annotated.
    # The second argument to Annotated is our reducer function,
    # which tells LangGraph how to merge parallel updates.
    describer_state: Annotated[AgentState, _merge_dicts]
    summarizer_state: Annotated[AgentState, _merge_dicts] 
    prompt_state: Annotated[PromptState, _merge_dicts]
    search_state: Annotated[SearchAgentState, _merge_dicts]


config = Config()

llm = ChatOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.url,
            model=config.websearch.model_name,
            temperature=config.websearch.temperature,
            model_kwargs={"max_tokens": 16000},
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
    llm=llm,
    role_name="Ассистент",
    role_description="Ты профессиональный рерайтер",
    role_instructions=[
        "Тебе надо суммаризовать описание в краткую аннотацию",
        "Текст надо сокращать с умом, чтобы не потерять важную информацию",
    ],
    additional_context="Ты встроен в граф задач."
)

new_agregate_agent = RoleAgent(
    llm=llm,
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


class Configuration(TypedDict):
    """Configurable parameters for the agent."""
    # API key for the TavilySearch service
    tavily_api_key: str
    # OpenAI API key for ChatOpenAI
    openai_api_key: str
    # Model name for ChatOpenAI (e.g., "gpt-3.5-turbo")
    model_name: str


def open_table(root: RootState) -> RootState:
    data = pd.read_excel()

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
            "graphic_url":  graphic_url
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

graph = (
    StateGraph(RootState, config_schema=Configuration)
      .add_node("describer_node", run_load_and_process)
      .add_node("summarize_describer_node", run_summarize)
      .add_node("prompt_node", write_prompt)
      .add_node("search_node", run_search)
      .add_node("news_aggregate", run_news_summarize)  # единый агрегатный узел

      .add_edge("__start__", "describer_node")
      .add_edge("__start__", "prompt_node")

      .add_edge("describer_node",      "summarize_describer_node")
      .add_edge("summarize_describer_node", END)

      .add_edge("prompt_node", "search_node")
      .add_edge("search_node", "news_aggregate")
      .add_edge("news_aggregate",  END)
      .compile(name="Parallel Two‑State Graph with Aggregate Summary")
)


