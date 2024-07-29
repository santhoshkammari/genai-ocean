import numpy as np
# tool_selector.py
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from torch import cosine_similarity, argsort

from tool_registry import ToolRegistry
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json


class ToolSelector:
    def __init__(self, llm: BaseChatModel, tool_registry: ToolRegistry):
        self.tool_embeddings = None
        self.llm = llm
        self.tool_registry = tool_registry
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def select_tool(self, user_input: str,top_m = 5) -> dict:
        if self.tool_embeddings is None:
            self._generate_embeddings()
            

        # Generate embedding for user input
        user_embedding = self.model.encode([user_input],convert_to_tensor=True)
        # Calculate cosine similarity between user input and tool descriptions
        similarities = cosine_similarity(user_embedding, self.tool_embeddings)
        # Get indices of top 3 most similar tools
        top_3_indices = argsort(similarities, descending=True)[:top_m].tolist()
        print(top_3_indices)
        # Get top 3 most similar tools
        top_3_tools = [self.tools_info[i] for i in top_3_indices]


        system_message = self._create_system_message(top_3_tools)
        human_message = HumanMessage(content=f"User input: {user_input}")

        print(ChatPromptTemplate.from_messages([
            system_message,
            human_message
        ]).invoke({}).to_string())

        input_messgae = [system_message, human_message]

        response = self.llm.invoke(input_messgae)

        try:
            try:
                jp = JsonOutputParser()
                tool_selection = jp.parse(response.content)
            except:
                tool_selection = json.loads(response.content)

            if isinstance(tool_selection["arguments"], dict):
                return tool_selection
            else:
                return {
                    "selected_tool": tool_selection["selected_tool"],
                    "arguments": {"query": tool_selection["arguments"]}
                }
        except json.JSONDecodeError:
            return {"selected_tool": "None", "arguments": {}}

    def _format_parameters(self, param_model):
        if not param_model:
            return "None"

        params = []
        for field_name, field in param_model.__fields__.items():
            param_type = field.type_.__name__
            params.append(f"{field_name}: ({param_type})")
            # params.append(f"{field_name} ({param_type}): {param_description}")

        return ", ".join(params)

    def _create_system_message(self, tools_info: list) -> SystemMessage:
        tools_description = "\n".join([
            f"{tool['name']}: Description({tool['description']}) ## [parameters]: ({self._format_parameters(tool.get('param_model'))})"
            for tool in tools_info
        ])
        print(tools_description)
        return SystemMessage(content=f"""Tools:
        {tools_description}

        Respond in JSON:
         {{
            "selected_tool": "tool_name",
            "arguments": {{[parameters]}}
        }}""")

    def _generate_embeddings(self):
        self.tools_info = self.tool_registry.get_tools_info()
        tool_descriptions = [f"{tool['name']}: {tool['description']}" for tool in self.tools_info]
        self.tool_embeddings = self.model.encode(tool_descriptions,convert_to_tensor=True)

# tool_selector.py
# from langchain_core.language_models import BaseChatModel
# from tool_registry import ToolRegistry
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# import json
#
#
# class ToolSelector:
#     def __init__(self, llm: BaseChatModel, tool_registry: ToolRegistry):
#         self.llm = llm
#         self.tool_registry = tool_registry
#
#     def select_tool(self, user_input: str) -> dict:
#         tools_info = self.tool_registry.get_tools_info()
#         system_message = self._create_system_message(tools_info)
#         human_message = HumanMessage(content=f"User input: {user_input}")
#
#         input_messgae = [system_message, human_message]
#
#         response = self.llm.invoke(input_messgae)
#
#         try:
#             tool_selection = json.loads(response.content)
#             if isinstance(tool_selection["arguments"], dict):
#                 return tool_selection
#             else:
#                 return {
#                     "selected_tool": tool_selection["selected_tool"],
#                     "arguments": {"query": tool_selection["arguments"]}
#                 }
#         except json.JSONDecodeError:
#             return {"selected_tool": "None", "arguments": {}}
#
#     def _format_parameters(self, param_model):
#         if not param_model:
#             return "None"
#
#         params = []
#         for field_name, field in param_model.__fields__.items():
#             param_type = field.type_.__name__
#             params.append(f"{field_name}: ({param_type})")
#             # params.append(f"{field_name} ({param_type}): {param_description}")
#
#         return ", ".join(params)
#
#     def _create_system_message(self, tools_info: list) -> SystemMessage:
#         tools_description = "\n".join([
#             f"{tool['name']}: Description({tool['description']}) ## [parameters]: ({self._format_parameters(tool.get('param_model'))})"
#             for tool in tools_info
#         ])
#         print(tools_description)
#         return SystemMessage(content=f"""Tools:
#         {tools_description}
#
#         Respond in JSON:
#          {{
#             "selected_tool": "tool_name",
#             "arguments": {{[parameters]}}
#         }}""")
#
