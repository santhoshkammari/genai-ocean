from typing import List, Optional, Union, Any, Iterator
import requests
import torch
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, BaseMessageChunk
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.runnables import RunnableConfig
from pydantic import Field
from torch import dtype


class APIBasedChatLLM(BaseChatModel):
    model: str = Field(default="Qwen/Qwen2-0.5B-Instruct",description="Model Name")
    token: str = Field(default="",description="Huggingface Api Token")
    api_url: str = "http://localhost:8000"
    use_streaming = Field(default=False, alias="stream")
    max_new_tokens = Field(default=5000, alias="max_new_tokens")
    temperature = Field(default=1e-3, alias="temperature")
    top_p = Field(default=0.9, alias="top_p")
    torch_dtype: str = "float32"
    quantization: str = Field(default="", alias="quantization")

    model_kwargs: dict = Field(default_factory=dict)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._check_model_status()

    def _check_model_status(self):
        response = requests.get(f"{self.api_url}/model_status")
        if response.status_code == 200:
            status = response.json()
            if not status["loaded"]:
                self._initialize_model()
        else:
            raise Exception(f"Failed to check model status: {response.text}")


    def _initialize_model(self):
        response = requests.post(f"{self.api_url}/load_model", json={
            "model": self.model,
            "token": self.token,
            "torch_dtype": self.torch_dtype,
            "quantization": self.quantization
        })
        if response.status_code != 200:
            raise Exception(f"Failed to initialize model: {response.text}")

    @property
    def _llm_type(self):
        return "custom_chat"

    def _call_api(self, messages: List[dict]) -> str:
        __data = self._prepare_data(messages)
        result = ""

        with requests.post(f"{self.api_url}/chat", json=__data, stream=True) as response:
            if response.status_code == 200:
                for text in response.iter_content(chunk_size=None, decode_unicode=True):
                    result += text
            else:
                print(f"Error: {response.status_code}")
                print(response.text)

        return result

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs
    ) -> ChatResult:
        api_messages = self._prepare_messages(messages)
        content = self._call_api(api_messages)
        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _prepare_messages(self, messages: List[BaseMessage]) -> List[dict]:
        return [{"role": self._get_role(msg), "content": msg.content} for msg in messages]

    def _get_role(self, message: BaseMessage) -> str:
        if isinstance(message, HumanMessage):
            return "user"
        elif isinstance(message, AIMessage):
            return "assistant"
        else:
            return "system"

    def invoke(self, input: Union[str, List[BaseMessage]], **kwargs) -> AIMessage:
        if isinstance(input, str):
            messages = [HumanMessage(content=input)]
        else:
            messages = input
        return self._generate(messages, **kwargs).generations[0].message

    def stream(
            self,
            input: LanguageModelInput,
            config: Optional[RunnableConfig] = None,
            *,
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> Iterator[BaseMessageChunk]:

        if isinstance(input, str):
            messages = [HumanMessage(content=input)]
        else:
            messages = input
        api_messages = self._prepare_messages(messages)
        for chunk in self._stream_call_api(api_messages):
            yield AIMessage(content=chunk)

    def _stream_call_api(self, messages: List[dict]) -> Iterator[str]:
        __data = self._prepare_data(messages)

        with requests.post(f"{self.api_url}/chat", json=__data, stream=True) as response:
            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    yield chunk
            else:
                print(f"Error: {response.status_code}")
                print(response.text)

    def _prepare_data(self, messages: List[dict]) -> dict:
        return dict(
            messages=messages,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            **self.model_kwargs
        )
