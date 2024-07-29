import sys
from typing import Dict, List, Any, Optional, TextIO, Iterator
from langchain.chains.base import Chain
from langchain_core.runnables import RunnableConfig
from langchain_core.utils import print_text
from pydantic import BaseModel, Field

from hermes_thoth import HermesThoth
from tool_registry import ToolRegistry
from tool_selector import ToolSelector
from tool_executor import ToolExecutor
from result_processor import ResultProcessor
from response_generator import ResponseGenerator


class ToolCallingChain(Chain, BaseModel):
    llm: HermesThoth
    tool_registry: ToolRegistry = Field(default=None)
    tool_selector: ToolSelector = Field(default=None)
    tool_executor: ToolExecutor = Field(default=None)
    result_processor: ResultProcessor = Field(default=None)
    response_generator: ResponseGenerator = Field(default=None)
    verbose: bool = Field(default=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.tool_registry is None:
            self.tool_registry = self.llm.tool_registry
        if self.tool_selector is None:
            self.tool_selector = ToolSelector(self.llm, self.tool_registry)
        if self.tool_executor is None:
            self.tool_executor = ToolExecutor(self.tool_registry)
        if self.result_processor is None:
            self.result_processor = ResultProcessor(self.llm)
        if self.response_generator is None:
            self.response_generator = ResponseGenerator(self.llm)

    @property
    def input_keys(self) -> List[str]:
        return ["input"]

    @property
    def output_keys(self) -> List[str]:
        return ["output"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        output = ""
        for chunk in self._stream(inputs):
            output += chunk
        return {"output": output}

    def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._call(inputs=input)

    def _stream(self, inputs: Dict[str, Any]) -> Iterator[str]:
        if self.verbose:
            print_text("\n\n/----------\\\n", color="green")
            print_text("| ToolCallingChain |\n", color="green")
            print_text("\\----------/\n\n", color="green")

        user_input = inputs["input"]
        if self.verbose:
            print_text("User Input: ", color="blue")
            print_text(f"{user_input}\n\n")

        selected_tool_info = self.tool_selector.select_tool(user_input)
        if self.verbose:
            print_text("Selected Tool: ", color="yellow")
            print_text(f"{selected_tool_info['selected_tool']}\n")
            if selected_tool_info["arguments"]:
                print_text("Arguments: ", color="yellow")
                print_text(f"{selected_tool_info['arguments']}\n\n")

        if selected_tool_info["selected_tool"] == "None":
            if self.verbose:
                print_text("No tool selected. Generating response...\n", color="red")
            for chunk in self.response_generator.stream(user_input, ""):
                yield chunk
            return

        tool_result = self.tool_executor.execute(selected_tool_info["selected_tool"], selected_tool_info["arguments"])
        if self.verbose:
            print_text("Tool Execution Result: ", color="pink")
            print_text(f"{tool_result}\n\n")

        # chat_prompt_value = self.result_processor.chat_prompt.invoke({
        #     "input": user_input,
        #     "tool_name": selected_tool_info["selected_tool"],
        #     "tool_result": str(tool_result)
        # }).to_string()

        for c in self.result_processor.process(input=user_input,
                                               tool_name=selected_tool_info["selected_tool"],
                                               tool_result=tool_result):
            if self.verbose:
                print_text(c.content, end="", color="green")
            yield c.content

    def stream(self, input: Dict[str, Any], **kwargs: Any) -> Iterator[Dict[str, str]]:
        for chunk in self._stream(input):
            yield {"output": chunk}