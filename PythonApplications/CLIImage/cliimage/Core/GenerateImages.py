class GenerateImages:
    def __init__(self, app):
        self._app = app

    def generate_image(self, prompt_index: int = 0):

        if len(self._app._flux_nunchaku_and_loras._prompt_embeds) == 0:
            self._app._flux_nunchaku_and_loras.create_prompt_embeds()

        if prompt_index >= len(self._app._flux_nunchaku_and_loras._prompt_embeds):
            self._app._terminal_ui.print_error(
                "Prompt index is greater than the number of prompt embeds")
            return False

        self._app._flux_nunchaku_and_loras.delete_text_encoder_2_and_pipeline()
        self._app._flux_nunchaku_and_loras.create_transformer_and_pipeline()

        self._app._flux_nunchaku_and_loras.update_transformer_with_loras()

        images = \
            self._app._flux_nunchaku_and_loras.call_pipeline_with_prompt_embed(
                prompt_index)

        batch_processing_configuration = \
            self._app._process_configurations.get_batch_processing_configuration()

        batch_processing_configuration.create_and_save_image(
            0,
            images[0],
            self._app._flux_nunchaku_and_loras._generation_configuration,
            self._app._process_configurations.get_model_name())

        self._app._terminal_ui.print_success("Image generated successfully!")

        return True

    def generate_depth_image(
            self,
            prompt_index: int = 0,
            control_image_index: int = 0):
        if len(self._app._flux_depth_nunchaku_and_loras._control_images) == 0:
            self._app._flux_depth_nunchaku_and_loras.create_control_image()

        if control_image_index >= len(
            self._app._flux_depth_nunchaku_and_loras._control_images):
            self._app._terminal_ui.print_error(
                "Control image index is greater than the number of control images")
            return False

        self._app._flux_depth_nunchaku_and_loras._delete_processor()

        if len(self._app._flux_depth_nunchaku_and_loras._prompt_embeds) == 0:
            self._app._flux_depth_nunchaku_and_loras.create_prompt_embeds()

        if prompt_index >= len(self._app._flux_depth_nunchaku_and_loras._prompt_embeds):
            self._app._terminal_ui.print_error(
                "Prompt index is greater than the number of prompt embeds")
            return False

        self._app._flux_depth_nunchaku_and_loras._delete_text_encoder_2_and_pipeline()
        self._app._flux_depth_nunchaku_and_loras.create_transformer_and_pipeline()
        self._app._flux_depth_nunchaku_and_loras.update_transformer_with_loras()

        images = \
            self._app._flux_depth_nunchaku_and_loras.call_pipeline(
                prompt_index,
                control_image_index)

        batch_processing_configuration = \
            self._app._process_configurations.get_batch_processing_configuration()

        batch_processing_configuration.create_and_save_image(
            0,
            images[0],
            self._app._flux_depth_nunchaku_and_loras._generation_configuration,
            self._app._process_configurations.get_model_name())

        self._app._terminal_ui.print_success(
            "Image with depth control generated successfully!")

        return True

    def generate_kontext_image(
            self,
            prompt_index: int = 0,
            control_image_index: int = 0):
        if len(self._app._flux_kontext_nunchaku_and_loras._control_images) == 0:
            self._app._flux_kontext_nunchaku_and_loras.load_control_image()

        if control_image_index >= len(
            self._app._flux_kontext_nunchaku_and_loras._control_images):
            self._app._terminal_ui.print_error(
                "Control image index is greater than the number of control images")
            return False

        if len(self._app._flux_kontext_nunchaku_and_loras._prompt_embeds) == 0:
            self._app._flux_kontext_nunchaku_and_loras.create_prompt_embeds()

        if prompt_index >= len(self._app._flux_kontext_nunchaku_and_loras._prompt_embeds):
            self._app._terminal_ui.print_error(
                "Prompt index is greater than the number of prompt embeds")
            return False

        self._app._flux_kontext_nunchaku_and_loras._delete_text_encoder_2_and_pipeline()
        self._app._flux_kontext_nunchaku_and_loras.create_transformer_and_pipeline()
        self._app._flux_kontext_nunchaku_and_loras.update_transformer_with_loras()

        images = \
            self._app._flux_kontext_nunchaku_and_loras.call_pipeline(
                prompt_index,
                control_image_index)

        batch_processing_configuration = \
            self._app._process_configurations.get_batch_processing_configuration()

        batch_processing_configuration.create_and_save_image(
            0,
            images[0],
            self._app._flux_depth_nunchaku_and_loras._generation_configuration,
            self._app._process_configurations.get_model_name())

        self._app._terminal_ui.print_success(
            "Image using kontext generated successfully!")

        return True

    def process_batch(self, prompt_index: int = 0):
        batch_processing_configuration = \
            self._app._process_configurations.get_batch_processing_configuration()

        for index in range(batch_processing_configuration.number_of_images):
            images = \
                self._app._flux_nunchaku_and_loras.call_pipeline_with_prompt_embed(
                    prompt_index)
            if images is not None:
                batch_processing_configuration.create_and_save_image(
                        index,
                        images[0],
                        self._app._flux_nunchaku_and_loras._generation_configuration,
                        self._app._process_configurations.get_model_name())
            else:
                self._app._terminal_ui.print_error(
                    "Pipeline execution failed! Images is None")

            self._app._flux_nunchaku_and_loras._generation_configuration.guidance_scale += \
                batch_processing_configuration.guidance_scale_step

        self._app._terminal_ui.print_success(
            "Batch processing completed successfully!")

    def process_batch_depth_images(self, prompt_index: int = 0, control_image_index: int = 0):
        batch_processing_configuration = \
            self._app._process_configurations.get_batch_processing_configuration()

        for index in range(batch_processing_configuration.number_of_images):
            images = \
                self._app._flux_depth_nunchaku_and_loras.call_pipeline(
                    prompt_index,
                    control_image_index)
            if images is not None:
                batch_processing_configuration.create_and_save_image(
                    index,
                    images[0],
                    self._app._flux_depth_nunchaku_and_loras._generation_configuration,
                    self._app._process_configurations.get_model_name())
            else:
                self._app._terminal_ui.print_error(
                    "Pipeline execution failed! Images is None")

            self._app._flux_depth_nunchaku_and_loras._generation_configuration.guidance_scale += \
                batch_processing_configuration.guidance_scale_step

        self._app._terminal_ui.print_success(
            "Batch processing for depth images completed successfully!")

        return True

    def process_batch_kontext_images(self, prompt_index: int = 0, control_image_index: int = 0):
        batch_processing_configuration = \
            self._app._process_configurations.get_batch_processing_configuration()

        for index in range(batch_processing_configuration.number_of_images):
            images = \
                self._app._flux_kontext_nunchaku_and_loras.call_pipeline(
                    prompt_index,
                    control_image_index)
            if images is not None:
                batch_processing_configuration.create_and_save_image(
                    index,
                    images[0],
                    self._app._flux_kontext_nunchaku_and_loras._generation_configuration,
                    self._app._process_configurations.get_model_name())
            else:
                self._app._terminal_ui.print_error(
                    "Pipeline execution failed! Images is None")

            self._app._flux_kontext_nunchaku_and_loras._generation_configuration.guidance_scale += \
                batch_processing_configuration.guidance_scale_step

        self._app._terminal_ui.print_success(
            "Batch processing for kontext images completed successfully!")

        return True

    def process_batch_over_all_nunchaku_models(self, prompt_index: int = 0):
        if len(self._app._flux_nunchaku_and_loras._prompt_embeds) == 0:
            self._app._flux_nunchaku_and_loras.create_prompt_embeds()

        if prompt_index >= len(self._app._flux_nunchaku_and_loras._prompt_embeds):
            self._app._terminal_ui.print_error(
                "Prompt index is greater than the number of prompt embeds")
            return False

        self._app._flux_nunchaku_and_loras.delete_text_encoder_2_and_pipeline()

        nunchaku_model_paths = \
            self._app._process_configurations.configurations[
                "nunchaku_configuration"].nunchaku_model_paths

        batch_processing_configuration = \
            self._app._process_configurations.get_batch_processing_configuration()

        for nunchaku_model_index, nunchaku_model_path in enumerate(
            nunchaku_model_paths):
            self._app._flux_nunchaku_and_loras.create_transformer_and_pipeline(
                nunchaku_model_index)

            self._app._flux_nunchaku_and_loras.refresh_configurations(
                self._app._process_configurations.configurations[
                    "nunchaku_configuration"],
                self._app._process_configurations.configurations[
                    "flux_generation_configuration"],
                self._app._process_configurations.configurations[
                    "pipeline_inputs"],
                self._app._process_configurations.configurations[
                    "nunchaku_loras_configuration"])

            self._app._flux_nunchaku_and_loras.update_transformer_with_loras()

            for index in range(batch_processing_configuration.number_of_images):
                images = \
                    self._app._flux_nunchaku_and_loras.call_pipeline_with_prompt_embed(
                        prompt_index)
                if images is not None:
                    batch_processing_configuration.create_and_save_image(
                        index,
                        images[0],
                        self._app._flux_nunchaku_and_loras._generation_configuration,
                        self._app._process_configurations.get_model_name(
                            nunchaku_model_index))
                else:
                    self._app._terminal_ui.print_error(
                        "Pipeline execution failed! Images is None")

                self._app._flux_nunchaku_and_loras._generation_configuration.guidance_scale += \
                    batch_processing_configuration.guidance_scale_step

            self._app._terminal_ui.print_info(
                f"Completed batch processing for nunchaku model {nunchaku_model_path}")

            self._app._flux_nunchaku_and_loras.delete_transformer_and_pipeline()

        self._app._terminal_ui.print_success(
            "Batch processing over all nunchaku models completed successfully!")

        return True