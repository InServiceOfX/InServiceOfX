from moretransformers.Tools import ToolCallProcessor

from typing import Callable

class ToolsManager:
    def __init__(self, available_functions=None):
        self._tool_call_processor = ToolCallProcessor(
            available_functions=available_functions)

    def add_tool_call(tool_name: str, tool: Callable):
        self._tool_call_processor.add_function(tool_name, tool)

    def get_tool_names(self):
        return self._tool_call_processor.available_functions.keys()