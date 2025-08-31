from cliimage.Core import ProcessConfigurations
from cliimage.Terminal import CommandHandler
from cliimage.Terminal import PromptSessionsManager
from cliimage.Terminal import TerminalUI

from morediffusers.Applications import FluxNunchakuAndLoRAs

class CLIImage:
    def __init__(self, application_paths):
        self._application_paths = application_paths
        self._terminal_ui = TerminalUI()

        self._process_configurations = ProcessConfigurations(
            application_paths,
            self._terminal_ui)

        self._process_configurations.process_configurations()

        self._flux_nunchaku_and_loras = FluxNunchakuAndLoRAs(
            self._process_configurations.configurations[
                "nunchaku_configuration"],
            self._process_configurations.configurations[
                "flux_generation_configuration"],
            self._process_configurations.configurations["pipeline_inputs"],
            self._process_configurations.configurations[
                "nunchaku_loras_configuration"])

        self._command_handler = CommandHandler(self)

        self._prompt_sessions_manager = PromptSessionsManager(self)

    def run_iterative(self):
        try:
            prompt = self._prompt_sessions_manager.prompt(
                "Image generation prompt (or type .help for options): "
            )

            if not prompt.strip():
                return True

            if prompt.startswith('.'):
                continue_running, command_handled = \
                    self._command_handler.handle_command(prompt)

                if not command_handled:
                    self._terminal_ui.print_processing(prompt)

                return continue_running

            # Generate response
            self._terminal_ui.print_processing(prompt)
            return True

        except KeyboardInterrupt:
            self._terminal_ui.print_goodbye()
            return False
        except Exception as e:
            self._terminal_ui.print_error(str(e))
            return True

    def run(self):
        self._terminal_ui.print_header("CLIImage - Image Generation Tool")
        continue_running = True
        while continue_running:
            continue_running = self.run_iterative()
