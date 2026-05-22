from typing import Optional
from warnings import warn

import torch

class GenerateImages:
    def __init__(self, app):
        self._app = app

    def _log_nunchaku_generation(
        self,
        model_index: int,
        generation_hash: Optional[str] = None,
        truncated_generation_hash: Optional[str] = None) -> None:
        self._app._process_logs.log_nunchaku_generation(
            self._app._flux_nunchaku_and_loras._configuration,
            self._app._flux_nunchaku_and_loras._generation_configuration,
            self._app._flux_nunchaku_and_loras._pipeline_inputs,
            self._app._flux_nunchaku_and_loras._loras_configuration,
            model_index,
            generation_hash,
            truncated_generation_hash)

    def generate_image(self, prompt_index: int = 0):

        if len(self._app._flux_nunchaku_and_loras._prompt_embeds) == 0:
            self._app._flux_nunchaku_and_loras.create_prompt_embeds()

        if prompt_index >= len(self._app._flux_nunchaku_and_loras._prompt_embeds):
            self._app._terminal_ui.print_error(
                "Prompt index is greater than the number of prompt embeds")
            return False

        self._app._flux_nunchaku_and_loras.delete_text_encoder_2_and_pipeline()
        # By default, this sets nunchaku_model_index to 0
        self._app._flux_nunchaku_and_loras.create_transformer_and_pipeline()

        self._app._flux_nunchaku_and_loras.update_transformer_with_loras()

        images = \
            self._app._flux_nunchaku_and_loras.call_pipeline_with_prompt_embed(
                prompt_index)

        batch_processing_configuration = \
            self._app._process_configurations.get_batch_processing_configuration()

        full_hash, config_hash = \
            batch_processing_configuration.create_and_save_image(
                0,
                images[0],
                self._app._flux_nunchaku_and_loras._generation_configuration,
                self._app._process_configurations.get_model_name())

        try:
            self._log_nunchaku_generation(
                model_index=0,
                generation_hash=full_hash,
                truncated_generation_hash=config_hash)
        except Exception as e:
            warn(f"Warning: Could not log nunchaku generation: {e}")

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
            self._app._process_configurations.get_control_model_name())

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
        """
        Run batch image generation, sweeping guidance_scale across all images.

        Self-contained: loads the model and creates prompt embeddings if they
        are not already in memory, so this can be called directly without
        running generate_image() first.  Running generate_image() beforehand
        is still useful as a quick preview before committing to a longer batch.
        """
        flux = self._app._flux_nunchaku_and_loras

        # Create prompt embeddings if not already done.  This loads the text
        # encoder, encodes the prompt, then the encoder is freed from VRAM
        # inside create_prompt_embeds() before the transformer loads.
        if len(flux._prompt_embeds) == 0:
            self._app._terminal_ui.print_info(
                "Creating prompt embeddings...")
            flux.create_prompt_embeds()

        if prompt_index >= len(flux._prompt_embeds):
            self._app._terminal_ui.print_error(
                "Prompt index is greater than the number of prompt embeds")
            return False

        # Load the transformer + pipeline if not already in VRAM.
        if not flux.is_transformer_enabled():
            self._app._terminal_ui.print_info(
                "Loading model pipeline (first run — this takes a minute)...")
            flux.delete_text_encoder_2_and_pipeline()
            flux.create_transformer_and_pipeline()
            flux.update_transformer_with_loras()

        batch_processing_configuration = \
            self._app._process_configurations.get_batch_processing_configuration()

        for index in range(batch_processing_configuration.number_of_images):
            images = flux.call_pipeline_with_prompt_embed(prompt_index)
            if images is not None:
                full_hash, config_hash = \
                    batch_processing_configuration.create_and_save_image(
                        index,
                        images[0],
                        flux._generation_configuration,
                        self._app._process_configurations.get_model_name())

                try:
                    self._log_nunchaku_generation(
                        model_index=0,
                        generation_hash=full_hash,
                        truncated_generation_hash=config_hash)
                except Exception as e:
                    warn(f"Warning: Could not log nunchaku generation: {e}")
            else:
                self._app._terminal_ui.print_error(
                    "Pipeline execution failed! Images is None")

            flux._generation_configuration.guidance_scale += \
                batch_processing_configuration.guidance_scale_step

            # Free cached VRAM allocations between images.
            torch.cuda.empty_cache()

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

            self._app._process_configurations.process_configurations()

            print("Guidance scale before refresh: ",
                self._app._process_configurations.configurations[
                "flux_generation_configuration"].guidance_scale)

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
                    full_hash, config_hash = \
                        batch_processing_configuration.create_and_save_image(
                            index,
                            images[0],
                            self._app._flux_nunchaku_and_loras._generation_configuration,
                            self._app._process_configurations.get_model_name(
                                nunchaku_model_index))

                    try:
                        self._log_nunchaku_generation(
                            model_index=nunchaku_model_index,
                            generation_hash=full_hash,
                            truncated_generation_hash=config_hash)
                    except Exception as e:
                        warn(f"Warning: Could not log nunchaku generation: {e}")
                else:
                    self._app._terminal_ui.print_error(
                        "Pipeline execution failed! Images is None")

                self._app._flux_nunchaku_and_loras._generation_configuration.guidance_scale += \
                    batch_processing_configuration.guidance_scale_step

                # Free cached VRAM allocations between images so subsequent
                # generations don't OOM on memory-constrained GPUs (e.g. 8 GB).
                torch.cuda.empty_cache()

            self._app._terminal_ui.print_info(
                f"Completed batch processing for nunchaku model {nunchaku_model_path}")

            self._app._flux_nunchaku_and_loras.delete_transformer_and_pipeline()

        self._app._terminal_ui.print_success(
            "Batch processing over all nunchaku models completed successfully!")

        return True