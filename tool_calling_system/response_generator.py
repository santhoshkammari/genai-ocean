# response_generator.py
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

class ResponseGenerator:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ('human',"""Given the following user input and processed result, generate a final response to the user.

User input: {input}
Processed result: {processed_result}

Final response:"""
             )])

    def generate(self, input: str, processed_result: str) -> str:
        prompt_value = self.prompt.invoke({"input": input, "processed_result": processed_result})
        for chunk in self.llm.stream(prompt_value.messages):
            yield chunk.content
        # return self.llm.invoke(prompt_value.messages)