from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from pydantic import BaseModel, Field
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
import json
from langchain.output_parsers import PydanticOutputParser


class SearchResult(BaseModel):
    title: str
    content: str
    url: str
    relevance_score: float = 0.0
    relevance_reasoning: str = ""

class RelevanceEvaluation(BaseModel):
    relevance_score: float = Field(ge=0, le=10, description="Relevance score from 0 to 10")
    reasoning: str = Field(description="Brief explanation of the relevance score")

class SearchAgent:

    def __init__(self, config):
        self._llm = ChatOpenAI(
                api_key=config.api_key.get_secret_value(),
                base_url=config.url,
                model=config.websearch.model_name,
                temperature=config.websearch.temperature,
                model_kwargs={"max_tokens": config.websearch.max_tokens},
                extra_body={"use_beam_search": config.websearch.use_beam_search, "best_of": config.websearch.best_of}
            )
        
        self._search_tool = TavilySearch(max_results=5, topic='general')
        
    #     self.agent = self._init_agent()

    # def _init_agent(self):
    #     return create_react_agent(self._llm, [self._search_with_rerank])
    
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
                raise Exception(e)
        
        reranked_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return reranked_results
    
    def _format_results(self, results: List[SearchResult]) -> str:
        """Форматирует результаты для вывода"""
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
    

    def search_with_rerank(self, query: str, top_k: int = 10):
        """Выполняет поиск и реранжирование результатов"""
        raw_result = self._search_tool.invoke(query)
        reranked_results = self._rerank_results(raw_result['results'], query)

        relevant_results = [r for r in reranked_results if r.relevance_score >= 5.0]

        if not reranked_results:
            return self._format_results(reranked_results[:top_k])
        return self._format_results(relevant_results[:top_k])
    

    