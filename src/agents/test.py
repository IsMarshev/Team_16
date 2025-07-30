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

class SearchResult(BaseModel):
    title: str
    content: str
    url: str
    relevance_score: float = 0.0
    relevance_reasoning: str = ""

class EnhancedWebSearchAgent:
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
        
        self._reranker_llm = ChatOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.url,
            model=config.websearch.model_name,
            temperature=config.websearch.temperature,
            model_kwargs={"max_tokens": config.websearch.max_tokens}
        )
        
        self._search_tool = TavilySearch(max_results=10)
        
        self._enhanced_search_tool = Tool(
            name="enhanced_web_search",
            func=self._search_with_reranking,
            description="Search the web and rerank results based on relevance to the query"
        )
        
        self.agent = self._init_agent()
        
    def _init_agent(self):
        # Улучшенный промпт для агента
        system_prompt = """You are a web search assistant that finds and analyzes information.

                When searching:
                1. First, analyze the user's query to understand the context and domain
                2. Use the enhanced_web_search tool to find relevant information
                3. The tool will return reranked results based on relevance
                4. Synthesize the most relevant information to answer the user's question

                Always think about the domain context when evaluating results."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        agent = create_react_agent(
            self._llm, 
            [self._enhanced_search_tool], 
            prompt = prompt
        )
        return agent
        
        return AgentExecutor(
            agent=agent,
            tools=[self._enhanced_search_tool],
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    def _extract_domain_context(self, query: str) -> Dict[str, Any]:
        """Извлекает контекст и домен из запроса"""
        context_prompt = f"""Analyze this search query and extract:
                1. Main domain/industry (e.g., aviation, technology, medicine)
                2. Key concepts and entities
                3. Search intent

                Query: {query}

                Return as JSON with keys: domain, key_concepts, intent"""
        
        response = self._reranker_llm.invoke([
            SystemMessage(content="You are a query analyzer. Return only valid JSON."),
            HumanMessage(content=context_prompt)
        ])
        
        try:
            return json.loads(response.content)
        except:
            return {
                "domain": "general",
                "key_concepts": [query],
                "intent": "information seeking"
            }
    
    def _rerank_results(self, results: List[Dict], query: str, context: Dict[str, Any]) -> List[SearchResult]:
        """Реранжирует результаты поиска на основе релевантности"""
        reranked_results = []
        
        reranking_prompt_template = """Evaluate the relevance of this search result to the query.
            Query: {query}
            Domain/Industry: {domain}
            Key Concepts: {key_concepts}

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
                domain=context.get("domain", "general"),
                key_concepts=", ".join(context.get("key_concepts", [])),
                title=result.get("title", ""),
                content=result.get("content", "")[:500]
            )
            
            response = self._reranker_llm.invoke([
                SystemMessage(content="You are a relevance evaluator. Return only valid JSON."),
                HumanMessage(content=prompt)
            ])
            
            try:
                evaluation = json.loads(response.content)
                search_result = SearchResult(
                    title=result.get("title", ""),
                    content=result.get("content", ""),
                    url=result.get("url", ""),
                    relevance_score=float(evaluation.get("relevance_score", 0)),
                    relevance_reasoning=evaluation.get("reasoning", "")
                )
                reranked_results.append(search_result)
            except Exception as e:
                # Если не удалось распарсить, добавляем с низким скором
                search_result = SearchResult(
                    title=result.get("title", ""),
                    content=result.get("content", ""),
                    url=result.get("url", ""),
                    relevance_score=0.0,
                    relevance_reasoning=f"Error during evaluation: {str(e)}"
                )
                reranked_results.append(search_result)
        
        # Сортируем по релевантности
        reranked_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return reranked_results
    
    def _search_with_reranking(self, query: str) -> str:
        """Выполняет поиск и реранжирование результатов"""
        context = self._extract_domain_context(query)
        
        raw_results = self._search_tool.run(query)
        print(raw_results)
        
        if isinstance(raw_results, str):
            try:
                results_list = json.loads(raw_results)
            except:
                results_list = [{"title": "Result", "content": raw_results, "url": ""}]
        else:
            results_list = raw_results if isinstance(raw_results, list) else [raw_results]
        
        reranked = self._rerank_results(results_list, query, context)
        
        relevant_results = [r for r in reranked if r.relevance_score >= 5.0]
        
        if not relevant_results:
            return "No highly relevant results found. Here are the best matches:\n\n" + \
                   self._format_results(reranked[:3])
        
        return f"Found {len(relevant_results)} relevant results:\n\n" + \
               self._format_results(relevant_results[:5])
    
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
    
    def search(self, query: str) -> str:
        """Выполняет поиск с реранжированием (не stream-режим)"""
        enhanced_query = f"""Search Query: {query}

        Пожалуйста, найдите соответствующую информацию и предоставьте исчерпывающий ответ на основе результатов повторного ранжирования."""
        
        # Предполагается, что у self.agent есть метод, возвращающий результат целиком, например .run()
        response = self.agent.invoke(
            {"messages": [HumanMessage(content=enhanced_query)]}
        )
        # # Извлекаем текст ответа
        # if response["messages"]:
        #     return response["messages"][-1].content
        # return ""
        return response