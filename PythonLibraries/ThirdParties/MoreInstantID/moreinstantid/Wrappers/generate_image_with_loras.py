from moreinstantid.Wrappers import generate_image

def generate_image_with_loras(
    pipe,
    prompt,
    face_information,
    prompt_2=None,
    negative_prompt=None,
    negative_prompt_2=None,
    pose_information=None,
    ip_adapter_scale=1.0,
    controlnet_conditioning_scale=1.0,
    number_of_steps=50,
    guidance_scale=None,
    lora_scale=None,
    clip_skip=None,
    generator=None,
    ):
    if lora_scale == None:

        return generate_image(
            pipe,
            prompt,
            face_information,
            prompt_2,
            negative_prompt,
            negative_prompt_2,
            pose_information,
            ip_adapter_scale,
            controlnet_conditioning_scale,
            number_of_steps,
            guidance_scale,
            clip_skip,
            generator)

    else:

        keypoints = None
        width = None
        height = None
        if pose_information == None:
            keypoints = face_information.face_keypoints
            height = face_information.height
            width = face_information.width
        else:
            keypoints = pose_information.pose_keypoints
            height = pose_information.height
            width = pose_information.width

        if negative_prompt == None:
            negative_prompt = ""

        pipe.set_ip_adapter_scale(ip_adapter_scale)

        if guidance_scale == None:
            # __call__() will return a type
            # 'diffusers.pipelines.stable_diffusion_xl.pipeline_output.StableDiffusionXLPipelineOutput'
            # and can be index by number, yielding a list.
            # The class member .images is also a list, a list of PIL.Image.Image.
            images = pipe(
                prompt=prompt,
                prompt_2=prompt_2,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                image_embeds=face_information.face_embedding,
                image=keypoints,
                controlnet_conditioning_scale=float(controlnet_conditioning_scale),
                num_inference_steps=number_of_steps,
                height=height,
                width=width,
                generator=generator,
                cross_attention_kwargs={"scale": float(lora_scale)},
                clip_skip=clip_skip
                ).images

        else:
            """
            Recall that in
            diffusers/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl.py
            in StableDiffusionXLControlNetPipeline.__call__(..),
            which has the same implementation (for code related to guidance_scale)
            as in InstantID/pipeline_stable_diffusion_xl_instantid.py
            guidance_scale input has default value 5.0 and is a property (uses
            @property decorator).
            is used if self.unet.config.time_cond_proj_dim is not None (I've found
            that for most of the diffusion models used, this is None for the unet)
            and if self.do_classifier_free_guidance (which is found to be true for

            so that
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond)

            and noise_pred used here,
            # compute the previous noise sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
                **extra_step_kwargs,
                return_dict=False)[0]
            """
            images = pipe(
                prompt=prompt,
                prompt_2=prompt_2,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                image_embeds=face_information.face_embedding,
                image=keypoints,
                controlnet_conditioning_scale=float(controlnet_conditioning_scale),
                num_inference_steps=number_of_steps,
                guidance_scale=float(guidance_scale),
                height=height,
                width=width,
                generator=generator,
                cross_attention_kwargs={"scale": float(lora_scale)},
                clip_skip=clip_skip,
                ).images

        return images[0]