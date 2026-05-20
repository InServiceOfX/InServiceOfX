export interface LoraConfig {
  nickname: string;
  directory_path: string;
  filename: string;
  lora_strength: number;
  is_active: boolean;
  description?: string;
}

export interface ProfileSummary {
  name: string;
  complete: boolean;
  missing: string[];
}

export interface CliImageConfig {
  batch: Record<string, unknown> & {
    number_of_images?: number;
    base_filename?: string;
    guidance_scale_step?: number;
  };
  flux: Record<string, unknown> & {
    width?: number;
    height?: number;
    num_inference_steps?: number;
    guidance_scale?: number;
    true_cfg_scale?: number;
    temporary_save_path?: string;
  };
  nunchaku: Record<string, unknown>;
  control: Record<string, unknown>;
  loras: Record<string, unknown> & {
    lora_scale?: number;
    loras?: LoraConfig[];
  };
  pipeline: Record<string, unknown> & {
    prompt?: string;
    prompt_2?: string;
    negative_prompt?: string;
    negative_prompt_2?: string;
  };
  profiles: ProfileSummary[];
}

export interface CliImageStatus {
  cuda_device?: string;
  flux_model_path?: string;
  nunchaku_model_count?: number;
  width?: number;
  height?: number;
  steps?: number;
  guidance_scale?: number;
  true_cfg_scale?: number;
  output_path?: string;
  batch_images?: number;
  prompt?: string;
  negative_prompt?: string;
  active_lora_count?: number;
  active_loras?: LoraConfig[];
}

export interface ConfigLocation {
  config_dir: string;
  default_config_dir: string;
}
