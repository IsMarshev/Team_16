from typing import Dict, Any, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Состояние для графа"""
    messages: List[BaseMessage]
    context: Optional[Dict[str, Any]]


class RoleAgent:
    """Ролевой агент для использования в LangGraph"""
    
    def __init__(
        self, 
        llm: BaseChatModel,
        role_name: str,
        role_description: str,
        role_instructions: List[str],
        additional_context: Optional[str] = None
    ):
        self.llm = llm
        self.role_name = role_name
        self.role_description = role_description
        self.role_instructions = role_instructions
        self.additional_context = additional_context
        
        # Создаем системный промпт для роли
        self.system_prompt = self._create_system_prompt()
        
        # Создаем шаблон промпта
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        # Создаем цепочку
        self.chain = self.prompt_template | self.llm
    
    def _create_system_prompt(self) -> str:
        """Создает системный промпт на основе параметров роли"""
        prompt_parts = [
            f"Ты - {self.role_name}.",
            f"Описание роли: {self.role_description}",
            "\nТвои инструкции:"
        ]
        
        for i, instruction in enumerate(self.role_instructions, 1):
            prompt_parts.append(f"{i}. {instruction}")
        
        if self.additional_context:
            prompt_parts.append(f"\nДополнительный контекст: {self.additional_context}")
        
        return "\n".join(prompt_parts)
    
    def __call__(self, state: AgentState) -> AgentState:
        """Вызов агента как узла графа"""
        messages = state["messages"]
        
        if state.get("context"):
            context_str = "\n".join([f"{k}: {v}" for k, v in state["context"].items()])
            modified_prompt = ChatPromptTemplate.from_messages([
                ("system", f"{self.system_prompt}\n\nТекущий контекст:\n{context_str}"),
                MessagesPlaceholder(variable_name="messages")
            ])
            chain = modified_prompt | self.llm
            response = chain.invoke({"messages": messages})
        else:
            response = self.chain.invoke({"messages": messages})
        
        return {
            "messages": messages + [response],
            "context": state.get("context")
        }

