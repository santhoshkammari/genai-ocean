# custom_chat_ollama.py
import json
import time
from typing import List, Dict, Any, Callable, Optional, Union

from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.pydantic_v1 import Field

from hermes import HermesLLM
from tool_registry import ToolRegistry
from huggingface_hub import InferenceClient


class MessageRole:
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool-call"
    TOOL_RESPONSE = "tool-response"


class HermesThoth(ChatOllama):
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    tool_registry: ToolRegistry = Field(default_factory=ToolRegistry)
    host_type: str = "ollama"
    hf_client: Optional[InferenceClient] = None

    def __init__(self, host_type: str = "ollama", **kwargs):
        super().__init__(**kwargs)
        self.host_type = host_type
        if self.host_type == "hf":
            self.hf_client = InferenceClient(model=self.model, timeout=120)

    def bind_tools(self, tools: List[Union[Callable, BaseTool]]):
        self.tools = []
        for tool in tools:
            formatted_tool = self.format_tool(tool)
            self.tools.append(formatted_tool)
            self.register_tool(tool, formatted_tool)
        return self

    def register_tool(self, tool: Union[Callable, BaseTool], formatted_tool: Dict[str, Any]):
        if isinstance(tool, BaseTool):
            self.tool_registry.register_tool(tool, name=tool.name, description=tool.description)
        else:
            self.tool_registry.register_tool(tool, name=formatted_tool["name"],
                                             description=formatted_tool["description"])

    @staticmethod
    def format_tool(tool: Callable) -> Dict[str, Any]:
        if isinstance(tool, BaseTool):
            return {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        else:
            annotations = tool.__annotations__
            return {
                "name": tool.__name__,
                "description": tool.__doc__,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param: {"type": "string"} for param in annotations if param != "return"
                    },
                    "required": [param for param in annotations if param != "return"]
                }
            }

    def invoke(self, messages: Union[str, List[Dict[str, Any]]], **kwargs):
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]

        if self.host_type == "ollama":
            ollama_messages = self._convert_messages_to_ollama_messages(messages)
            response = super().invoke(ollama_messages, **kwargs)
            print(response)
            print('################################')
        elif self.host_type == "hf":
            hf_messages = self._convert_messages_to_hf_messages(messages)
            print(hf_messages)
            response = self._invoke_hf(hf_messages, **kwargs)
            print(response)
            print('################################')
        else:
            raise ValueError(f"Unsupported host_type: {self.host_type}")

        return response

    def _convert_messages_to_ollama_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        ollama_messages = []
        for message in messages:
            if isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
                role = "user" if isinstance(message, HumanMessage) else "assistant" if isinstance(message,
                                                                                                  AIMessage) else "system"
                ollama_messages.append({"role": role, "content": message.content})
            elif isinstance(message, dict) and "type" in message:
                role = "user" if message["type"] == "human" else "assistant" if message["type"] == "ai" else "system"
                ollama_messages.append({"role": role, "content": message["data"]["content"]})
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
        return ollama_messages

    def _convert_messages_to_hf_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        hf_messages = []
        for message in messages:
            if isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
                role = MessageRole.USER if isinstance(message, HumanMessage) else MessageRole.ASSISTANT if isinstance(
                    message, AIMessage) else MessageRole.SYSTEM
                hf_messages.append({"role": role, "content": message.content})
            elif isinstance(message, dict) and "type" in message:
                role = MessageRole.USER if message["type"] == "human" else MessageRole.ASSISTANT if message[
                                                                                                        "type"] == "ai" else MessageRole.SYSTEM
                hf_messages.append({"role": role, "content": message["data"]["content"]})
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
        return hf_messages

    def _invoke_hf(self, messages: List[Dict[str, str]], **kwargs) -> str:
        stop_sequences = kwargs.get("stop", [])
        response = self.hf_client.chat_completion(messages, stop=stop_sequences, max_tokens=1500)
        response = response.choices[0].message.content

        for stop_seq in stop_sequences:
            if response[-len(stop_seq):] == stop_seq:
                response = response[:-len(stop_seq)]

        return AIMessage(content=response)











# # custom_chat_ollama.py
# import json
# import time
# from typing import List, Dict, Any, Callable, Optional, Union
#
# from langchain_core.tools import BaseTool
# from langchain_ollama import ChatOllama
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# from langchain_core.pydantic_v1 import Field
#
# from hermes import HermesLLM
# from tool_registry import ToolRegistry
#
#
# class HermesThoth(ChatOllama):
#     tools: List[Dict[str, Any]] = Field(default_factory=list)
#     tool_registry: ToolRegistry = Field(default_factory=ToolRegistry)
#
#     def bind_tools(self, tools: List[Union[Callable, BaseTool]]):
#         self.tools = []
#         for tool in tools:
#             formatted_tool = self.format_tool(tool)
#             self.tools.append(formatted_tool)
#             self.register_tool(tool, formatted_tool)
#         return self
#
#     def register_tool(self, tool: Union[Callable, BaseTool], formatted_tool: Dict[str, Any]):
#         if isinstance(tool, BaseTool):
#             self.tool_registry.register_tool(tool, name=tool.name, description=tool.description)
#         else:
#             self.tool_registry.register_tool(tool, name=formatted_tool["name"],
#                                              description=formatted_tool["description"])
#
#     @staticmethod
#     def format_tool(tool: Callable) -> Dict[str, Any]:
#         if isinstance(tool, BaseTool):
#             return {
#                 "name": tool.name,
#                 "description": tool.description,
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "query": {"type": "string"}
#                     },
#                     "required": ["query"]
#                 }
#             }
#         else:
#             annotations = tool.__annotations__
#             return {
#                 "name": tool.__name__,
#                 "description": tool.__doc__,
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         param: {"type": "string"} for param in annotations if param != "return"
#                     },
#                     "required": [param for param in annotations if param != "return"]
#                 }
#             }
#
#
#
#     def invoke(self, messages: Union[str, List[Dict[str, Any]]],**kwargs):
#         if isinstance(messages,str):
#             messages = [HumanMessage(content=messages)]
#         ollama_messages = self._convert_messages_to_ollama_messages(messages)
#         response = super().invoke(ollama_messages, **kwargs)
#         return response
#
#     def _convert_messages_to_ollama_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
#         ollama_messages = []
#         for message in messages:
#             if isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
#                 role = "user" if isinstance(message, HumanMessage) else "assistant" if isinstance(message, AIMessage) else "system"
#                 ollama_messages.append({"role": role, "content": message.content})
#             elif isinstance(message, dict) and "type" in message:
#                 role = "user" if message["type"] == "human" else "assistant" if message["type"] == "ai" else "system"
#                 ollama_messages.append({"role": role, "content": message["data"]["content"]})
#             else:
#                 raise ValueError(f"Unsupported message type: {type(message)}")
#         return ollama_messages
