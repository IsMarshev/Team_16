from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from src.agents import SearchAgent, SearchAgentState
from typing import Optional
from src.config import Config
from dotenv import load_dotenv

load_dotenv()


config = Config()
"""Создает граф с несколькими агентами, включая поисковый"""
    
# Создаем агентов
search_agent = SearchAgent(config)

# Агент-координатор


# Создаем граф
workflow = StateGraph(SearchAgentState)

# Добавляем узлы
workflow.add_node("search", search_agent)

# Настраиваем переходы
workflow.add_edge(START, "search")

# workflow.add_conditional_edges(
#     lambda x: x["next_action"],
#     {
#         "search": "search",
#         "respond": END,
#         "end": END
#     }
# )


# Компилируем граф
app = workflow.compile()

app.get_graph()
