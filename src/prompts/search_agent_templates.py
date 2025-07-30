from langchain.chains import MapReduceDocumentsChain, LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="Сделай краткое резюме и выдели 3 главных факта из этого текста:\n{text}"
)

final_prompt = PromptTemplate(
    input_variables=["summaries"],
    template="Сравни эти резюме, выдели общие и противоречивые моменты, сделай вывод:\n{summaries}"
)

summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

# Map-Reduce цепочка для всех результатов
map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=summary_chain,
    combine_prompt=final_prompt
)