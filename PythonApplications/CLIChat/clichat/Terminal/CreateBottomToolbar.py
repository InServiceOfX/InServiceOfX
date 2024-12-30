from prompt_toolkit.formatted_text import HTML

class CreateBottomToolbar:
    def __init__(self, configuration, runtime_configuration):
        self._configuration = configuration
        self._runtime_configuration = runtime_configuration

    def create_bottom_toolbar(self):
        def _create():
            history_status = (
                "【User History: ON】" 
                if self._runtime_configuration.is_user_prompt_history_active 
                else "<dim>【User History: OFF】</dim>"
            )
            return HTML(
                f"Type a message or command (<style fg='ansiyellow'>.help</style> for commands) "
                f"{history_status}"
                f""" {str(self._configuration.hotkey_exit).replace("'", "")} {self._configuration.exit_entry}"""
            )
        return _create
