
# result_processor.py
from typing import Any

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from hermes_thoth import HermesThoth


class ResultProcessor:
    def __init__(self, llm: BaseChatModel):
        self.llm:HermesThoth = llm
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("human",
             "Given the following user input, tool used, and tool result, provide a human-friendly direct precise answer of the result with nice styling formats. Include all relevant information, especially video titles and links.\n\nUser input: {input}\nTool used: {tool_name}\nTool result: {tool_result}\n\nAnswer:")

        ])
        self.chain = self.llm

    def process(self, input: str, tool_name: str, tool_result: Any) -> str:
        chat_prompt_value = self.chat_prompt.invoke({
            "input": input,
            "tool_name": tool_name,
            "tool_result": str(tool_result)
        })
        if self.llm.host_type == "hf":
            res =  self.llm.invoke(chat_prompt_value.messages)
            yield res
        else:
            for chunk in self.chain.stream(chat_prompt_value.messages):
                print(chunk.content,end="",flush=True)
                yield chunk

# # result_processor.py
# from typing import Any
#
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain_core.language_models import BaseChatModel
# from langchain_core.messages import HumanMessage
# from langchain_core.prompts import ChatPromptTemplate
#
#
# class ResultProcessor:
#     def __init__(self, llm: BaseChatModel):
#         self.llm = llm
#         self.chat_prompt = ChatPromptTemplate.from_messages([
#             ("human",
#              "Given the following user input, tool used, and tool result, provide a human-friendly direct precise answer of the result with nice styling formats. Include all relevant information, especially video titles and links.\n\nUser input: {input}\nTool used: {tool_name}\nTool result: {tool_result}\n\nAnswer:")
#
#         ])
#         self.chain = self.llm
#
#     def process(self, input: str, tool_name: str, tool_result: Any) -> str:
#         chat_prompt_value = self.chat_prompt.invoke({
#             "input": input,
#             "tool_name": tool_name,
#             "tool_result": str(tool_result)
#         })
#         print('Streaming')
#         for chunk in self.chain.stream(chat_prompt_value.messages):
#             print(chunk.content,end="",flush=True)
#         return self.chain.invoke(chat_prompt_value.messages)
