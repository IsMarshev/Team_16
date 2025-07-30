from src.agents import SearchAgent
from src.config import Config
import os
from dotenv import load_dotenv

load_dotenv()

config = Config()

agent = SearchAgent(config)


print(agent.search_with_rerank('Что случилось с аэрофлотом 28.07.2025'))