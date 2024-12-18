class CreateBottomToolbar:
    def __init__(self, configuration):
        self._configuration = configuration

    def create_bottom_toolbar(self):
        return \
            f""" {str(self._configuration.hotkey_exit).replace("'", "")} {self._configuration.exit_entry}"""
