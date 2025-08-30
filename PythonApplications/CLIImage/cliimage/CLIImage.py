from cliimage.Core import ProcessConfigurations
from cliimage.Terminal import CommandHandler
from cliimage.Terminal import PromptSessionsManager

from morediffusers.Applications import FluxNunchakuAndLoRAs

# Move these functions to Terminal UI.
from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import HTML

class CLIImage:
    def __init__(self, application_paths):
        self._application_paths = application_paths

        self._process_configurations = ProcessConfigurations(
            application_paths)

        self._process_configurations.process_configurations()

        self._flux_nunchaku_and_loras = FluxNunchakuAndLoRAs(
            self._process_configurations.configurations[
                "nunchaku_configuration"],
            self._process_configurations.configurations[
                "generation_configuration"],
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
                    print(f"Processing prompt: {prompt}")

                return continue_running

            # Generate response
            print(f"Processing prompt: {prompt}")
            return True

        except KeyboardInterrupt:
            print_formatted_text(HTML("\n<ansigreen>Goodbye!</ansigreen>"))
            return False
        except Exception as e:
            print_formatted_text(HTML(f"\n<ansired>Error: {str(e)}</ansired>\n"))
            return True

    def run(self):
        print("Running CLIImage")
        continue_running = True
        while continue_running:
            continue_running = self.run_iterative()

        print_formatted_text(HTML("\n<ansigreen>Goodbye!</ansigreen>"))