from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from langgraph.types import Command, interrupt
import json




class SearchResult(BaseModel):
    title: str
    content: str
    url: str
    relevance_score: float = 0.0
    relevance_reasoning: str = ""


class RelevanceEvaluation(BaseModel):
    relevance_score: float = Field(ge=0, le=10, description="Relevance score from 0 to 10")
    reasoning: str = Field(description="Brief explanation of the relevance score")


class SearchAgentState(TypedDict):
    """Состояние для SearchAgent в графе"""
    search_query: Optional[str]
    search_results: Optional[List[SearchResult]]
    formatted_results: Optional[str]
    context: Optional[Dict[str, Any]]


class SearchAgent:
    """Поисковый агент для использования в LangGraph"""
    
    def __init__(self, config):
        self._llm = ChatOpenAI(
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
        
        self._search_tool = TavilySearch(max_results=5, topic='general')
        self.config = config
    


    def __call__(self, state: SearchAgentState) -> SearchAgentState:
        """Основной метод для вызова агента как узла графа"""
        # Извлекаем поисковый запрос из последнего сообщения или из состояния
        search_query = state.get("search_query")
    
        
        if not search_query:
            # Если запрос не найден, возвращаем состояние с сообщением об ошибке
            error_message = AIMessage(
                content="Не удалось определить поисковый запрос. Пожалуйста, укажите, что нужно найти."
            )
            return {
                "search_results": state["search_results"] + [error_message],
                "formatted_results": None
            }
        
        # Выполняем поиск
        top_k = state.get("context", {}).get("top_k", 10)
        search_results, formatted_results = self._perform_search(search_query, top_k)
        
        # Создаем ответное сообщение
        response_message = AIMessage(
            content=f"Результаты поиска по запросу '{search_query}':\n\n{formatted_results}"
        )
        
        # Обновляем состояние
        return {
            "search_query": state["search_results"] + [response_message],
            "search_results": search_results,
            "formatted_results": formatted_results
        }
    
    def _extract_search_query(self, message_content: str) -> str:
        """Извлекает поисковый запрос из сообщения пользователя"""
        # Простая эвристика - можно заменить на более сложную логику или LLM
        keywords = ["найди", "поищи", "search", "find", "искать", "информация о", "что такое"]
        
        content_lower = message_content.lower()
        for keyword in keywords:
            if keyword in content_lower:
                # Извлекаем часть после ключевого слова
                parts = content_lower.split(keyword, 1)
                if len(parts) > 1:
                    return parts[1].strip()
        
        # Если ключевые слова не найдены, используем все сообщение как запрос
        return message_content
    
    def _perform_search(self, query: str, top_k: int = 10) -> tuple[List[SearchResult], str]:
        """Выполняет поиск и реранжирование"""
        raw_result = self._search_tool.invoke(query)
        reranked_results = self._rerank_results(raw_result['results'], query)
        
        relevant_results = [r for r in reranked_results if r.relevance_score >= 5.0]
        
        if not relevant_results:
            results_to_return = reranked_results[:top_k]
        else:
            results_to_return = relevant_results[:top_k]
        
        formatted = self._format_results(results_to_return)
        return results_to_return, formatted
    
    def _rerank_results(self, results: List[Dict], query: str) -> List[SearchResult]:
        """Реранжирует результаты поиска на основе релевантности"""
        reranked_results = []
        
        reranking_prompt_template = """Evaluate the relevance of this search result to the query.
            Query: {query}

            Search Result:
            Title: {title}
            Content: {content}

            Rate the relevance from 0 to 10 and explain your reasoning.
            Consider:
            - Does the content match the domain/industry?
            - Are the key concepts present and used in the right context?
            - Would this answer the user's query?

            Return JSON with keys: relevance_score (0-10), reasoning (brief explanation)
            """
        
        for result in results:
            prompt = reranking_prompt_template.format(
                query=query,
                title=result["title"],
                content=result["content"][:500]
            )
            
            structured_llm = self._llm.with_structured_output(RelevanceEvaluation)
            evaluation = structured_llm.invoke([
                SystemMessage(content="You are a relevance evaluator. Analyze the content and return structured output."),
                HumanMessage(content=prompt)
            ])
            
            try:
                search_result = SearchResult(
                    title=result['title'],
                    content=result['content'],
                    url=result['url'],
                    relevance_score=float(evaluation.relevance_score),
                    relevance_reasoning=evaluation.reasoning
                )
                reranked_results.append(search_result)
            except Exception as e:
                print(f"Error processing result: {e}")
                continue
        
        reranked_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return reranked_results
    
    def _format_results(self, results: List[SearchResult]) -> str:
        """Форматирует результаты для вывода"""
        if not results:
            return "Релевантные результаты не найдены."
        
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"{i}. {result.title}\n"
                f"   Relevance: {result.relevance_score}/10\n"
                f"   Reasoning: {result.relevance_reasoning}\n"
                f"   Content: {result.content[:200]}...\n"
                f"   URL: {result.url}\n"
            )
        return "\n".join(formatted)