from typing import Callable, Dict, Type, Union, List

from langchain_core.tools import BaseTool
from pydantic import BaseModel, create_model


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Dict] = {}



    def get_tool(self, name: str) -> Dict:
        return self.tools.get(name)

    def get_tools_info(self) -> List[Dict[str, str]]:
        return [{"name": name, "description": tool["description"],"param_model":tool.get("param_model")} for name, tool in self.tools.items()]
    def register_tool(self, tool: Union[Callable, BaseTool],name: str = None,description: str = None):
        if name is None:
            if isinstance(tool,BaseTool):
                name = tool.name
            else:
                name = tool.__name__

        # print(f"Function name: {tool.__name__}")
        # print(f"Function docstring: {tool.__doc__}")
        # print(f"Function module: {tool.__module__}")
        # print(f"Function annotations: {tool.__annotations__}")
        # print(f"Function defaults: {tool.__defaults__}")
        # print(f"Function kwargs defaults: {tool.__kwdefaults__}")
        # print(f"Function code: {tool.__code__}")
        # print(f"Function globals: {tool.__globals__}")
        if isinstance(tool, BaseTool):
            self.tools[name] = {
                "type": "langchain_tool",
                "tool": tool,
                "description": description
            }
        else:
            params = tool.__annotations__
            return_type = params.pop('return', None)

            # Create a Pydantic model for the function parameters
            fields = {
                param_name: (param_type, ...) for param_name, param_type in params.items()
            }
            ParamModel = create_model(f"{name.capitalize()}Params", **fields)

            self.tools[name] = {
                "type": "custom_function",
                "name": name  if name else tool.__name__ ,
                "description": description if description else  tool.__doc__,
                "function": tool,
                "param_model": ParamModel,
                "return_type": return_type
            }

