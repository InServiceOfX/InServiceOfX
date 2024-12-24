from prompt_toolkit.shortcuts import radiolist_dialog
from prompt_toolkit.styles import Style
from prompt_toolkit import HTML
from typing import Optional, Tuple
from moregroq.Wrappers import GetAllActiveModels
from corecode.Utilities import get_environment_variable

class ModelSelector:
    def __init__(self, configuration):
        self.style = Style.from_dict({
            'dialog': f'bg:{configuration.terminal_DialogBackgroundColor}',
            'dialog.body': f'bg:{configuration.terminal_DialogBackgroundColor}',
            'dialog frame.label': f'bg:{configuration.terminal_DialogBackgroundColor}'
        })

    def _get_available_models(self) -> list[tuple[str, str]]:
        try:
            get_all_active_models = GetAllActiveModels(
                api_key=get_environment_variable("GROQ_API_KEY"))
            get_all_active_models()
            models = get_all_active_models.get_list_of_available_models()
            return [
                (model.id, f"{model.id} (context window: {model.context_window})") 
                for model in models]
        except Exception as err:
            print(err)
            return [("llama-3.3-70b-versatile", 
                    "llama-3.3-70b-versatile (context window: 32768)")]

    def select_model_and_tokens(self) -> Tuple[str, Optional[int]]:
        models = self._get_available_models()
        
        model_result = radiolist_dialog(
            title="Model Selection",
            text="Select an LLM model:",
            values=models,
            default=models[0][0] if models else "llama-3.3-70b-versatile",
            style=self.style
        ).run()

        if not model_result:
            return "llama-3.3-70b-versatile", None

        token_values = [
            ("None", "Use model's default"),
            ("1024", "1024 tokens"),
            ("8192", "8192 tokens"),
            ("32768", "32768 tokens"),
            ("131072", "131072 tokens"),
        ]

        max_tokens_result = radiolist_dialog(
            title="Max Tokens Configuration",
            text=f"Select max tokens for {model_result}:",
            values=token_values,
            default="None",
            style=self.style
        ).run()

        max_tokens = None if max_tokens_result == "None" else int(
            max_tokens_result)
        return model_result, max_tokens