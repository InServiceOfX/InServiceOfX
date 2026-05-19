import shlex


class CommandHandler:
    def __init__(self, app):
        self._app = app
        # Command descriptions for completion and help
        self._command_descriptions = {
            ".exit": "Exit the application",
            ".help": "Show help message",
            ".batch_process_on_single_prompt": \
                "Intended to be run *after* running generate_image. Batch process on single prompt",
            ".batch_process_on_single_image_with_depth_control": \
                "Batch process on single image with depth control",
            ".batch_process_on_single_image_with_kontext": \
                "Batch process on single image with kontext control",
            ".batch_process_over_all_nunchaku_models": \
                "Batch process over all nunchaku models",
            ".generate_image": "Generate single image",
            ".generate_depth_image": \
                "Generate single image using depth control",
            ".generate_kontext_image": "Generate single image using kontext",
            ".list_loras": "List configured Nunchaku LoRAs",
            ".active_loras": "List active Nunchaku LoRAs",
            ".enable_lora": "Enable a LoRA by nickname",
            ".disable_lora": "Disable a LoRA by nickname",
            ".toggle_lora": "Toggle a LoRA by nickname",
            ".set_lora_strength": "Set a LoRA strength by nickname",
            ".refresh_configurations": "Refresh configurations",
            ".restart_all": "Restart all",
            "._update_with_loras": "Update the transformer with LoRAs",
        }

        self.commands = {
            ".exit": self.handle_exit,
            ".help": self.handle_help,
            ".batch_process_on_single_prompt": \
                self.handle_batch_process_on_single_prompt,
            ".batch_process_on_single_image_with_depth_control": \
                self.handle_batch_process_on_single_image_with_depth_control,
            ".batch_process_on_single_image_with_kontext": \
                self.handle_batch_process_on_single_image_with_kontext,
            ".batch_process_over_all_nunchaku_models": \
                self.handle_batch_process_over_all_nunchaku_models,
            ".generate_image": self.handle_generate_image,
            ".generate_depth_image": self.handle_generate_depth_image,
            ".generate_kontext_image": self.handle_generate_kontext_image,
            ".list_loras": self.handle_list_loras,
            ".active_loras": self.handle_active_loras,
            ".enable_lora": self.handle_enable_lora,
            ".disable_lora": self.handle_disable_lora,
            ".toggle_lora": self.handle_toggle_lora,
            ".set_lora_strength": self.handle_set_lora_strength,
            ".refresh_configurations": self.handle_refresh_configurations,
            ".restart_all": self.handle_restart_all,
            "._update_with_loras": self._handle__update_with_loras,
        }

        assert self._command_descriptions.keys() == self.commands.keys()

    def get_command_descriptions(self):
        return self._command_descriptions

    def handle_command(self, command: str) -> tuple[bool, bool]:
        """
        Handle a command and return whether to continue running and if command
        was handled.

        Args:
            command: The command string (including the dot prefix)

        Returns:
            tuple: (continue_running, command_handled)
                - continue_running: True to continue running, False to exit
                - command_handled: True if command was handled, False if it
                should be treated as user input
        """
        command = command.strip()
        command_parts = shlex.split(command)

        if not command_parts:
            return True, True

        command_name = command_parts[0].lower()
        args = command_parts[1:]

        if command_name in self.commands:
            return self.commands[command_name](args), True
        else:
            return True, False

    def handle_exit(self, args: list[str] | None = None) -> bool:
        self._app._terminal_ui.print_goodbye()
        return False

    def handle_help(self, args: list[str] | None = None) -> bool:
        # Generate help text from command descriptions
        help_lines = ["Available commands:"]
        for command, description in self._command_descriptions.items():
            # Format: command (padded to 25 chars) - description
            help_lines.append(f"  {command:<25} - {description}")
        
        help_text = "\n".join(help_lines)
        self._app._terminal_ui.print_help(help_text)
        return True

    def handle_batch_process_on_single_prompt(
            self,
            args: list[str] | None = None) -> bool:
        self._app._terminal_ui.print_info(
            "This is intended to be run *after* running generate_image. Batch processing on single prompt...")
    
        self._app._generate_images.process_batch()

        self.handle_refresh_configurations()

        return True

    def handle_batch_process_on_single_image_with_depth_control(
            self,
            args: list[str] | None = None) -> bool:
        self._app._terminal_ui.print_info(
            "Batch processing on single image with depth control...")

        self._app._generate_images.process_batch_depth_images()

        self.handle_refresh_configurations()

        return True

    def handle_batch_process_on_single_image_with_kontext(
            self,
            args: list[str] | None = None) -> bool:
        self._app._terminal_ui.print_info(
            "Batch processing on single image with kontext control...")

        self._app._generate_images.process_batch_kontext_images()

        self.handle_refresh_configurations()

        return True

    def handle_batch_process_over_all_nunchaku_models(
            self,
            args: list[str] | None = None) -> bool:
        self._app._terminal_ui.print_info(
            "Batch processing over all nunchaku models...")

        self._app._generate_images.process_batch_over_all_nunchaku_models()

        self.handle_refresh_configurations()

        return True

    def handle_refresh_configurations(self, args: list[str] | None = None) -> bool:
        self._app._terminal_ui.print_info("Refreshing configurations...")

        self._app._process_configurations.process_configurations()

        self._app._flux_nunchaku_and_loras.refresh_configurations(
            self._app._process_configurations.configurations[
                "nunchaku_configuration"],
            self._app._process_configurations.configurations[
                "flux_generation_configuration"],
            self._app._process_configurations.configurations["pipeline_inputs"],
            self._app._process_configurations.configurations[
                "nunchaku_loras_configuration"])

        if self._app._flux_kontext_nunchaku_and_loras is not None:
            self._app._flux_kontext_nunchaku_and_loras.refresh_configurations(
                self._app._process_configurations.configurations[
                    "nunchaku_configuration"],
                self._app._process_configurations.configurations[
                    "flux_generation_configuration"],
                self._app._process_configurations.configurations["pipeline_inputs"],
                self._app._process_configurations.configurations[
                    "nunchaku_loras_configuration"])

        if self._app._flux_depth_nunchaku_and_loras is not None:
            self._app._flux_depth_nunchaku_and_loras.refresh_configurations(
                self._app._process_configurations.configurations[
                    "nunchaku_flux_control_configuration"],
                self._app._process_configurations.configurations[
                    "flux_generation_configuration"],
                self._app._process_configurations.configurations["pipeline_inputs"],
                self._app._process_configurations.configurations[
                    "nunchaku_loras_configuration"])

        self._app._terminal_ui.print_success(
            "Configurations refreshed successfully!")
        return True

    def handle_generate_image(self, args: list[str] | None = None) -> bool:
        self._app._terminal_ui.print_info("Generating single image...")
        self._app._generate_images.generate_image()
        return True

    def handle_generate_depth_image(self, args: list[str] | None = None) -> bool:
        self._app._terminal_ui.print_info(
            "Generating single image using depth control...")
        self._app._generate_images.generate_depth_image()
        return True

    def handle_generate_kontext_image(self, args: list[str] | None = None) -> bool:
        self._app._terminal_ui.print_info(
            "Generating single image using kontext...")
        self._app._generate_images.generate_kontext_image()
        return True

    def _get_loras_configuration(self):
        return self._app._process_configurations.configurations[
            "nunchaku_loras_configuration"]

    def _get_loras_configuration_path(self):
        return self._app._application_paths.configuration_file_paths[
            "nunchaku_loras_configuration"]

    def _save_loras_configuration(self) -> None:
        self._get_loras_configuration().save_yaml(
            self._get_loras_configuration_path())

    def _require_lora_name(self, args: list[str]) -> str | None:
        if not args:
            self._app._terminal_ui.print_error("LoRA nickname is required")
            return None

        return " ".join(args)

    def _set_lora_active(self, nickname: str, is_active: bool) -> bool:
        loras_configuration = self._get_loras_configuration()

        if nickname not in loras_configuration.loras:
            self._app._terminal_ui.print_error(f"LoRA not found: {nickname}")
            return False

        loras_configuration.loras[nickname].is_active = is_active
        self._save_loras_configuration()
        state = "enabled" if is_active else "disabled"
        self._app._terminal_ui.print_success(f"LoRA {state}: {nickname}")
        return True

    def _format_lora_line(self, nickname, lora_parameters) -> str:
        state = "active" if lora_parameters.is_active else "inactive"
        return (
            f"  [{state:<8}] {nickname} "
            f"(strength={lora_parameters.lora_strength})")

    def handle_list_loras(self, args: list[str] | None = None) -> bool:
        loras_configuration = self._get_loras_configuration()

        if not loras_configuration.loras:
            self._app._terminal_ui.print_info("No LoRAs configured")
            return True

        lines = ["Configured LoRAs:"]
        for nickname, lora_parameters in loras_configuration.loras.items():
            lines.append(self._format_lora_line(nickname, lora_parameters))

        self._app._terminal_ui.print_info("\n".join(lines))
        return True

    def handle_active_loras(self, args: list[str] | None = None) -> bool:
        loras_configuration = self._get_loras_configuration()
        active_loras = loras_configuration.get_active_loras()

        if not active_loras:
            self._app._terminal_ui.print_info("No active LoRAs")
            return True

        lines = ["Active LoRAs:"]
        for nickname, lora_parameters in active_loras.items():
            lines.append(self._format_lora_line(nickname, lora_parameters))

        self._app._terminal_ui.print_info("\n".join(lines))
        return True

    def handle_enable_lora(self, args: list[str] | None = None) -> bool:
        nickname = self._require_lora_name(args or [])
        if nickname is None:
            return True

        self._set_lora_active(nickname, True)
        return True

    def handle_disable_lora(self, args: list[str] | None = None) -> bool:
        nickname = self._require_lora_name(args or [])
        if nickname is None:
            return True

        self._set_lora_active(nickname, False)
        return True

    def handle_toggle_lora(self, args: list[str] | None = None) -> bool:
        nickname = self._require_lora_name(args or [])
        if nickname is None:
            return True

        loras_configuration = self._get_loras_configuration()
        try:
            new_state = loras_configuration.toggle_lora(nickname)
        except ValueError as exception:
            self._app._terminal_ui.print_error(str(exception))
            return True

        self._save_loras_configuration()
        state = "enabled" if new_state else "disabled"
        self._app._terminal_ui.print_success(f"LoRA {state}: {nickname}")
        return True

    def handle_set_lora_strength(self, args: list[str] | None = None) -> bool:
        args = args or []
        if len(args) < 2:
            self._app._terminal_ui.print_error(
                "Usage: .set_lora_strength <nickname> <strength>")
            return True

        nickname = " ".join(args[:-1])

        try:
            strength = float(args[-1])
            self._get_loras_configuration().set_lora_strength(
                nickname,
                strength)
        except ValueError as exception:
            self._app._terminal_ui.print_error(str(exception))
            return True

        self._save_loras_configuration()
        self._app._terminal_ui.print_success(
            f"LoRA strength set: {nickname} = {strength}")
        return True

    def _handle__update_with_loras(self, args: list[str] | None = None) -> bool:
        self._app._terminal_ui.print_info("Updating with LoRAs...")
        self._app._flux_nunchaku_and_loras.update_transformer_with_loras()
        self._app._terminal_ui.print_success("LoRAs updated successfully!")
        return True

    def handle_restart_all(self, args: list[str] | None = None) -> bool:
        self._app._terminal_ui.print_info("Restarting all...")

        if self._app._flux_nunchaku_and_loras is not None:
            self._app._flux_nunchaku_and_loras.restart()

        if self._app._flux_kontext_nunchaku_and_loras is not None:
            self._app._flux_kontext_nunchaku_and_loras.restart()

        if self._app._flux_depth_nunchaku_and_loras is not None:
            self._app._flux_depth_nunchaku_and_loras.restart()

        self._app._terminal_ui.print_success("All restarted successfully!")
        return True
