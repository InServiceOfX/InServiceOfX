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