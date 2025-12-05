from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

class BaseToolCallProcessor(ABC):

    @staticmethod
    def default_result_to_string(result: Any) -> str:
        if isinstance(result, dict):
            return json.dumps(result)
        elif isinstance(result, str):
            return result
        else:
            return str(result)

    def __init__(self, process_function_result: Optional[Callable] = None):
        self._available_functions: Dict[str, Callable] = None
        self._process_function_result = process_function_result

    def add_function(self, function_name: str, function: Callable):
        if self._available_functions is None:
            self._available_functions = {}

        self._available_functions[function_name] = function

    def change_process_function_result(self, process_function_result: Callable):
        self._process_function_result = process_function_result

    @abstractmethod
    def handle_possible_tool_calls(self, response_message: Any):
        pass