class GenerateWithChatterbox:
    def __init__(self, app):
        self._app = app

    def generate_with_chatterbox(self):
        if not self._app._chatterbox_tts_model.is_model_loaded():
            self._app._chatterbox_tts_model.load_model()

        save_path = self._app._chatterbox_tts_model.generate_and_save()

        self._app._terminal_ui.print_info(
            f"Output saved to {str(save_path)}"
        )

    def generate_multiple_with_chatterbox(self):
        if not self._app._chatterbox_tts_model.is_model_loaded():
            self._app._chatterbox_tts_model.load_model()

        save_paths = \
            self._app._chatterbox_tts_model.generate_and_save_for_text_file_paths()

        for save_path in save_paths:
            self._app._terminal_ui.print_info(
                f"Output saved to {str(save_path)}"
            )
