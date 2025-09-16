
class GenerateWithVibeVoice:
    def __init__(self, app):
        self._app = app

    def generate_with_vibe_voice(self):
        if not self._app._vvmp.is_processor_loaded():
            self._app._vvmp.load_processor()

        if not self._app._vvmp.is_model_loaded():
            self._app._vvmp.load_model()

        self._app._vvmp.process_inputs()
        self._app._vvmp.generate()
        save_path, _ = self._app._vvmp.process_and_save_output()

        self._app._terminal_ui.print_info(
            f"Output saved to {str(save_path)}"
        )
