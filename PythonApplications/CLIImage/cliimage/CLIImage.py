from cliimage.Core import (GenerateImages, ProcessConfigurations, ProcessLogs)
from cliimage.Terminal import (
    CommandHandler,
    PromptSessionsManager,
    TerminalUI)

from morediffusers.Applications import (
    FluxDepthNunchakuAndLoRAs,
    FluxKontextNunchakuAndLoRAs,
    FluxNunchakuAndLoRAs
    )

class CLIImage:
    def __init__(self, application_paths):
        self._application_paths = application_paths
        self._terminal_ui = TerminalUI()

        self._process_configurations = ProcessConfigurations(
            application_paths,
            self._terminal_ui)
        self._process_configurations.process_configurations()

        self._process_logs = ProcessLogs(application_paths)

        self._generate_images = GenerateImages(self)

        self._flux_nunchaku_and_loras = FluxNunchakuAndLoRAs(
            self._process_configurations.configurations[
                "nunchaku_configuration"],
            self._process_configurations.configurations[
                "flux_generation_configuration"],
            self._process_configurations.configurations["pipeline_inputs"],
            self._process_configurations.configurations[
                "nunchaku_loras_configuration"])

        self._flux_kontext_nunchaku_and_loras = FluxKontextNunchakuAndLoRAs(
            self._process_configurations.configurations[
                "nunchaku_configuration"],
            self._process_configurations.configurations[
                "flux_generation_configuration"],
            self._process_configurations.configurations["pipeline_inputs"],
            self._process_configurations.configurations[
                "nunchaku_loras_configuration"])

        if self._process_configurations.configurations[
            "nunchaku_flux_control_configuration"] is not None:
            self._flux_depth_nunchaku_and_loras = FluxDepthNunchakuAndLoRAs(
                self._process_configurations.configurations[
                    "nunchaku_flux_control_configuration"],
                self._process_configurations.configurations[
                    "flux_generation_configuration"],
                self._process_configurations.configurations["pipeline_inputs"],
                self._process_configurations.configurations[
                    "nunchaku_loras_configuration"])
        else:
            self._flux_depth_nunchaku_and_loras = None

        self._command_handler = CommandHandler(self)

        self._prompt_sessions_manager = None

    def run_iterative(self):
        try:
            if self._prompt_sessions_manager is None:
                self._prompt_sessions_manager = PromptSessionsManager(self)

            prompt = self._prompt_sessions_manager.prompt("> ")

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

    def run_command(self, command: str) -> tuple[bool, bool]:
        command = command.strip()
        if not command:
            return True, True

        if not command.startswith("."):
            command = f".{command}"

        continue_running, command_handled = \
            self._command_handler.handle_command(command)

        if not command_handled:
            self._terminal_ui.print_error(f"Unknown command: {command}")

        return continue_running, command_handled

    def run_commands(self, commands: list[str]) -> bool:
        self._terminal_ui.print_header("CLIImage - Image Generation Tool")

        for command in commands:
            continue_running, command_handled = self.run_command(command)
            if not command_handled:
                return False
            if not continue_running:
                return True

        return True

    def run(self):
        self._terminal_ui.print_header("CLIImage - Image Generation Tool")
        self._command_handler.handle_status()
        self._terminal_ui.print_quick_commands()
        continue_running = True
        while continue_running:
            continue_running = self.run_iterative()
