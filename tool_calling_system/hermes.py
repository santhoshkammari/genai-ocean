

import time
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Any, Iterator, Sequence, Tuple, AsyncIterator
import requests
from langchain_core.runnables.utils import Input, Output
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, BaseMessageChunk
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.runnables import RunnableConfig

def get_api_url(api_url):
    machines = {
        'gpu2': "http://192.168.162.147:8000"
    }
    if 'http' in api_url:
        return api_url
    return machines.get(api_url,"http://localhost:8000")


def get_model(model):
    models = {
        "qwen2": "Qwen/Qwen2-0.5B-Instruct",
        'llama3': "meta-llama/Meta-Llama-3-8B-Instruct"
    }
    return models.get(model,model)


class HermesLLM(BaseChatModel):
    model: str = "Qwen/Qwen2-0.5B-Instruct"
    token: str = ""
    api_url: str = "http://localhost:8000"
    use_streaming: bool = False
    max_new_tokens: int
    temperature: float = 1e-3
    top_p: float = 0.9
    torch_dtype: str
    quantization: str = ""
    model_kwargs: dict = {}

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.__update_attributes()
        self._check_model_status()

    def _check_model_status(self):
        print(self.model)
        try:
            response = requests.get(f"{self.api_url}/model_status")
            if response.status_code == 200:
                status = response.json()
                if not status["loaded"] or status["model_name"] != self.model:
                    self._initialize_model()
            else:
                raise Exception(f"Failed to check model status: {response.text}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error connecting to the API: {str(e)}")

    def _initialize_model(self):
        print("Clearing existing models and loading new model...")
        try:
            clear_response = requests.post(f"{self.api_url}/clear_models")
            if clear_response.status_code != 200:
                raise Exception(f"Failed to clear existing models: {clear_response.text}")

            print("Loading Model...")
            model_input = {
                "model": self.model,
                "token": self.token,
                "torch_dtype": self.torch_dtype,
                "quantization": self.quantization
            }
            response = requests.post(f"{self.api_url}/load_model", json=model_input)
            if response.status_code != 200:
                raise Exception(f"Failed to initialize model: {response.text}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error connecting to the API: {str(e)}")

    @property
    def _llm_type(self) -> str:
        return "custom_chat"

    def _call_api(self, messages: List[dict]) -> str:
        __data = self._prepare_data(messages)
        result = ""

        try:
            with requests.post(f"{self.api_url}/chat", json=__data, stream=True) as response:
                if response.status_code == 200:
                    for text in response.iter_content(chunk_size=None, decode_unicode=True):
                        result += text
                else:
                    raise Exception(f"API error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error connecting to the API: {str(e)}")

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

    def _prepare_data(self, messages: List[dict]) -> dict:
        return {
            "messages": messages,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            **self.model_kwargs
        }

    def __update_attributes(self):
        self.api_url = get_api_url(self.api_url)
        if self.model in ["llama3"]:
            self.quantization = "4bit"
        self.model = get_model(self.model)


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
        print(__data)
        print(self.api_url)
        with requests.post(f"{self.api_url}/chat", json=__data, stream=True) as response:
            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    yield chunk
            else:
                print(f"Error: {response.status_code}")
                print(response.text)

    def batch_as_completed(
        self,
        inputs: Sequence[Input],
        config: Optional[Union[RunnableConfig, Sequence[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> Iterator[Tuple[int, Union[Output, Exception]]]:

        prepared_inputs = [
            self._prepare_messages([HumanMessage(msg)])
            if isinstance(msg,str) else msg
            for msg in inputs
        ]

        batch_size = kwargs.get("batch_size")

        for batch_start in range(0, len(prepared_inputs), batch_size):
            batch = prepared_inputs[batch_start:batch_start + batch_size]
            try:
                results = self._batch_call_api(batch)
                for i, result in enumerate(results):
                    yield (batch_start + i, AIMessage(content=result))
            except Exception as e:
                if return_exceptions:
                    yield (batch_start, e)
                else:
                    raise

    def _batch_call_api(self, messages: List[dict]) -> List[str]:
        payload = {
            "batch":
            [
                dict(
                    messages=message,
                    max_new_tokens = self.max_new_tokens,
                    temperature = self.temperature,
                    top_p= self.top_p
                )
            for message in messages
            ]
        }
        print(payload)

        response = requests.post(f"{self.api_url}/batch_chat", json=payload)

        if response.status_code == 200:
            results = response.json()["responses"]
            return results
        else:
            print(f"Error in batch : {response.text}")
            return []


