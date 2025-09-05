class CommandHandler:
    def __init__(self, app):
        self._app = app
        # Command descriptions for completion and help
        self._command_descriptions = {
            ".exit": "Exit the application",
            ".help": "Show help message",
            ".batch_process_on_single_prompt": "Batch process on single prompt",
            ".batch_process_on_single_image_with_depth_control": \
                "Batch process on single image with depth control",
            ".batch_process_on_single_image_with_kontext": \
                "Batch process on single image with kontext control",
            ".generate_image": "Generate single image",
            ".generate_depth_image": \
                "Generate single image using depth control",
            ".generate_kontext_image": "Generate single image using kontext",
            ".refresh_configurations": "Refresh configurations",
            ".restart_all": "Restart all",
            "._call_pipeline": "Execute the pipeline with current embeds",
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
            ".generate_image": self.handle_generate_image,
            ".generate_depth_image": self.handle_generate_depth_image,
            ".generate_kontext_image": self.handle_generate_kontext_image,
            ".refresh_configurations": self.handle_refresh_configurations,
            ".restart_all": self.handle_restart_all,
            "._call_pipeline": self._handle__call_pipeline,
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
        command = command.strip().lower()

        if command in self.commands:
            return self.commands[command](), True
        else:
            return True, False

    def _handle__call_pipeline(self) -> bool:
        self._app._terminal_ui.print_info("Calling pipeline...")

        try:
            images = self._app._flux_nunchaku_and_loras.call_pipeline_with_prompt_embed(0)

            batch_processing_configuration = \
                self._app._process_configurations.get_batch_processing_configuration()

            if images is not None:
                batch_processing_configuration.create_and_save_image(
                    0,
                    images[0],
                    self._app._flux_nunchaku_and_loras._generation_configuration,
                    self._app._process_configurations.get_model_name())

                self._app._terminal_ui.print_success(
                    "Pipeline executed successfully!")
            else:
                self._app._terminal_ui.print_error("Pipeline execution failed!")

            return True

        except Exception as e:
            self._app._terminal_ui.print_error(f"Pipeline execution failed: {e}")
            return False

    def handle_exit(self) -> bool:
        self._app._terminal_ui.print_goodbye()
        return False

    def handle_help(self) -> bool:
        # Generate help text from command descriptions
        help_lines = ["Available commands:"]
        for command, description in self._command_descriptions.items():
            # Format: command (padded to 25 chars) - description
            help_lines.append(f"  {command:<25} - {description}")
        
        help_text = "\n".join(help_lines)
        self._app._terminal_ui.print_help(help_text)
        return True

    def handle_batch_process_on_single_prompt(self) -> bool:
        self._app._terminal_ui.print_info(
            "Batch processing on single prompt...")
    
        self._app._generate_images.process_batch()

        self.handle_refresh_configurations()

        return True

    def handle_batch_process_on_single_image_with_depth_control(self) -> bool:
        self._app._terminal_ui.print_info(
            "Batch processing on single image with depth control...")

        self._app._generate_images.process_batch_depth_images()

        self.handle_refresh_configurations()

        return True

    def handle_batch_process_on_single_image_with_kontext(self) -> bool:
        self._app._terminal_ui.print_info(
            "Batch processing on single image with kontext control...")

        self._app._generate_images.process_batch_kontext_images()

        self.handle_refresh_configurations()

        return True

    def handle_refresh_configurations(self) -> bool:
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

    def handle_generate_image(self) -> bool:
        self._app._terminal_ui.print_info("Generating single image...")
        self._app._generate_images.generate_image()
        return True

    def handle_generate_depth_image(self) -> bool:
        self._app._terminal_ui.print_info(
            "Generating single image using depth control...")
        self._app._generate_images.generate_depth_image()
        return True

    def handle_generate_kontext_image(self) -> bool:
        self._app._terminal_ui.print_info(
            "Generating single image using kontext...")
        self._app._generate_images.generate_kontext_image()
        return True

    def _handle__update_with_loras(self) -> bool:
        self._app._terminal_ui.print_info("Updating with LoRAs...")
        self._app._flux_nunchaku_and_loras.update_transformer_with_loras()
        self._app._terminal_ui.print_success("LoRAs updated successfully!")
        return True

    def handle_restart_all(self) -> bool:
        self._app._terminal_ui.print_info("Restarting all...")

        if self._app._flux_nunchaku_and_loras is not None:
            self._app._flux_nunchaku_and_loras.restart()

        if self._app._flux_kontext_nunchaku_and_loras is not None:
            self._app._flux_kontext_nunchaku_and_loras.restart()

        if self._app._flux_depth_nunchaku_and_loras is not None:
            self._app._flux_depth_nunchaku_and_loras.restart()

        self._app._terminal_ui.print_success("All restarted successfully!")
        return True