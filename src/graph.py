from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, TypedDict
from langgraph.graph import END



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



class RootState(TypedDict):
    # assistant_state: AgentState
    prompt_state:   AgentState
    search_state:   SearchAgentState


config = Config()

llm = ChatOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.url,
            model=config.websearch.model_name,
            temperature=config.websearch.temperature,
            model_kwargs={"max_tokens": config.websearch.max_tokens},
            extra_body={
                "use_beam_search": config.websearch.use_beam_search, 
                "best_of": config.websearch.best_of
            }
        )


assistant_agent = RoleAgent(
    llm=llm,
    role_name="Ассистент",
    role_description="Помогает пользователю с вопросами по Python",
    role_instructions=[
        "Отвечай развернуто и понятно.",
        "Если нужно – приводи примеры кода.",
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


def run_assistant(root: RootState) -> RootState:
    new_state = assistant_agent(root["assistant_state"])
    root["assistant_state"] = new_state
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
    return root




# Определяем граф
graph = (
    StateGraph(RootState, config_schema=Configuration)
      # узлы‑адаптеры
    # .add_node("assistant_node", run_assistant)
    .add_node("prompt_node",    write_prompt)
    .add_node("search_node",    run_search)

      # ветвим от старта на два агент‑узла
    # .add_edge("__start__", "assistant_node")
    .add_edge("__start__", "prompt_node")

      # обе ветки сходятся на поисковом узле
    # .add_edge("assistant_node", "search_node")
    .add_edge("prompt_node", "search_node")

    # .add_edge("search_node", "review_node")
      # завершаем
    .add_edge("search_node", END)
    .compile(name="Parallel Two‑State Graph")
)
