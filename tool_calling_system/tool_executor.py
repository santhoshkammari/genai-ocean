from typing import Any, Dict
from tool_registry import ToolRegistry

class ToolExecutor:
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry

    def execute(self, tool_name: str, args: Dict[str, Any]) -> Any:
        tool_info = self.tool_registry.get_tool(tool_name)
        if not tool_info:
            raise ValueError(f"Tool '{tool_name}' not found")

        if tool_info["type"] == "langchain_tool":
            tool = tool_info["tool"]
            # For langchain tools, we expect a single 'query' argument
            if 'query' not in args:
                query = ','.join(f"{k}={v}" for k, v in args.items())
            else:
                query = args['query']
            return tool.run(query)
        else:  # custom_function
            # For custom functions, use the previous logic
            # Validate and convert arguments using the Pydantic model
            param_model = tool_info["param_model"]
            validated_args = param_model(**args)
            # Execute the function with validated arguments
            return tool_info["function"](**validated_args.dict())