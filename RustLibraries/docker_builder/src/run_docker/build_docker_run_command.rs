use crate::configuration::run_docker_configuration::RunDockerConfigurationData;
use std::path::Path;

//------------------------------------------------------------------------------
/// Configuration used for the inputs into the function to build the docker run
/// command.
//------------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct BuildDockerRunCommandConfiguration {
    /// Docker image name to run (required)
    pub docker_image_name: String,

    /// Volumes and ports loaded from run_configuration.yml
    pub run_config: RunDockerConfigurationData,

    /// GPU device ID (for --gpus device=N)
    pub gpu_id: Option<u32>,

    /// Run in detached mode (-d)
    pub is_detached: bool,

    /// Run with interactive terminal (-it)
    pub is_interactive: bool,

    /// Custom entrypoint (--entrypoint)
    pub entrypoint: Option<String>,

    /// Use host network (--network host)
    pub use_host_network: bool,

    /// Additional networks to connect to
    pub networks: Vec<String>,

    /// Custom container name (--name)
    pub container_name: Option<String>,

    /// Enable GUI support (X11 forwarding)
    pub enable_gui: bool,

    /// Enable audio support (PulseAudio)
    pub enable_audio: bool,

    /// Additional environment variables
    pub env_vars: Vec<(String, String)>,
}

impl Default for BuildDockerRunCommandConfiguration {
    fn default() -> Self {
        Self {
            docker_image_name: String::new(),
            run_config: RunDockerConfigurationData::default(),
            gpu_id: None,
            is_detached: false,
            // Default to interactive
            is_interactive: true,
            entrypoint: None,
            use_host_network: false,
            networks: vec![],
            container_name: None,
            enable_gui: false,
            enable_audio: false,
            env_vars: vec![],
        }
    }
}

//------------------------------------------------------------------------------
/// Add X11 GUI support to docker run command. Clearly, this isn't needed nor
/// desired for a 'headless' container, and so is default false.
//------------------------------------------------------------------------------
fn add_gui_support(cmd: &mut Vec<String>) {
    let display = std::env::var("DISPLAY").unwrap_or_else(|_| ":0".to_string());
    cmd.push("-e".to_string());
    cmd.push(format!("DISPLAY={}", display));
    cmd.push("-v".to_string());
    cmd.push("/tmp/.X11-unix:/tmp/.X11-unix:rw".to_string());
}

//------------------------------------------------------------------------------
/// Add audio support (PulseAudio) to docker run command.
/// 
/// Mounts PulseAudio socket and sets environment variables for audio.
/// Clearly, this isn't needed nor desired for a 'headless' container, and so is
/// default false.
//------------------------------------------------------------------------------
fn add_audio_support(cmd: &mut Vec<String>) {
    // Get user ID (in Python: os.getuid())
    #[cfg(unix)]
    let user_id = {
        use nix::unistd::getuid;
        getuid().as_raw()
    };
    
    #[cfg(not(unix))]
    let user_id = 1000;

    let pulse_socket = format!("/run/user/{}/pulse", user_id);
    let pulse_native = format!("/run/user/{}/pulse/native", user_id);

    // Check if PulseAudio socket exists
    if Path::new(&pulse_native).exists() {
        // Mount PulseAudio socket
        cmd.push("-v".to_string());
        cmd.push(format!("{}:/run/user/1000/pulse:ro", pulse_socket));

        cmd.push("-e".to_string());
        cmd.push("PULSE_SERVER=unix:/run/user/1000/pulse/native".to_string());

        cmd.push("-e".to_string());
        cmd.push("PULSE_RUNTIME_PATH=/run/user/1000/pulse".to_string());

        // Find and mount the PulseAudio cookie for authentication.
        // Cookie can be in multiple locations.
        let cookie_paths = [
            format!(
                "{}/.config/pulse/cookie",
                std::env::var("HOME").unwrap_or_default()),
            format!("/run/user/{}/.config/pulse/cookie", user_id),
            format!("/run/user/{}/pulse-cookie", user_id),
        ];

        let mut cookie_mounted = false;
        for cookie_path in &cookie_paths {
            if std::path::Path::new(cookie_path).exists() {
                cmd.push("-v".to_string());
                cmd.push(format!(
                    "{}:/run/user/1000/pulse-cookie:ro",
                    cookie_path));
                cookie_mounted = true;
                break;
            }
        }

        // Fallback: try cookie in socket directory
        if !cookie_mounted {
            let pulse_cookie_in_socket = format!("{}/cookie", pulse_socket);
            if std::path::Path::new(&pulse_cookie_in_socket).exists() {
                cmd.push("-v".to_string());
                cmd.push(format!(
                    "{}:/run/user/1000/pulse-cookie:ro",
                    pulse_cookie_in_socket));
            }
        }
    }

    // Always add ALSA device as fallback (Python does this)
    cmd.push("--device".to_string());
    cmd.push("/dev/snd".to_string());
}

pub fn build_docker_run_command(
    configuration: &BuildDockerRunCommandConfiguration,
) -> Result<Vec<String>, String> {
    if configuration.docker_image_name.is_empty() {
        return Err("Docker image name is empty".to_string());
    }

    let mut docker_run_cmd = vec!["docker".to_string(), "run".to_string()];

    // GPU support
    if let Some(gpu) = configuration.gpu_id {
        docker_run_cmd.push("--gpus".to_string());
        docker_run_cmd.push(format!("device={}", gpu));
    }

    if configuration.is_detached {
        docker_run_cmd.push("-d".to_string());
    }
    // Remove container after exit (unless detached)
    else
    {
        docker_run_cmd.push("--rm".to_string());
    }

    // Interactive + TTY
    if configuration.is_interactive {
        docker_run_cmd.push("-it".to_string());
    }

    // Network host option (if enabled, possibly skip port mappings)
    if configuration.use_host_network {
        docker_run_cmd.push("--network".to_string());
        docker_run_cmd.push("host".to_string());
    }

    // Additional networks
    for network in &configuration.networks {
        docker_run_cmd.push("--network".to_string());
        docker_run_cmd.push(network.clone());
    }

    // Add ports from config
    for port_map in &configuration.run_config.ports {
        docker_run_cmd.push("-p".to_string());
        docker_run_cmd.push(
            format!("{}:{}",
            port_map.host_port,
            port_map.container_port));
    }

    // Volume mounts from YAML config
    for volume in &configuration.run_config.volumes {
        docker_run_cmd.push("-v".to_string());
        docker_run_cmd.push(
            format!("{}:{}",
            volume.host_path,
            volume.container_path));
    }

    // GUI and audio support
    if configuration.enable_gui {
        add_gui_support(&mut docker_run_cmd);
    }
    if configuration.enable_audio {
        add_audio_support(&mut docker_run_cmd);
    }

    // Environment variables for NVIDIA runtime
    // Don't set CUDA_VISIBLE_DEVICES - let Docker handle GPU filtering
    docker_run_cmd.push("-e".to_string());
    docker_run_cmd.push("NVIDIA_DISABLE_REQUIRE=1".to_string());
    docker_run_cmd.push("-e".to_string());
    docker_run_cmd.push("CUDA_VISIBLE_DEVICES=0".to_string());

    // Runtime flags
    docker_run_cmd.push("--ipc=host".to_string());
    docker_run_cmd.push("--ulimit".to_string());
    docker_run_cmd.push("memlock=-1".to_string());
    docker_run_cmd.push("--ulimit".to_string());
    docker_run_cmd.push("stack=67108864".to_string());

    // Environment variables
    for (key, value) in &configuration.env_vars {
        docker_run_cmd.push("-e".to_string());
        docker_run_cmd.push(format!("{}={}", key, value));
    }

    // Container name
    if let Some(name) = &configuration.container_name {
        docker_run_cmd.push("--name".to_string());
        docker_run_cmd.push(name.clone());
    }

    // Entrypoint
    if let Some(entrypoint) = &configuration.entrypoint {
        docker_run_cmd.push("--entrypoint".to_string());
        docker_run_cmd.push(entrypoint.clone());
    }

    // Add image
    docker_run_cmd.push(configuration.docker_image_name.to_string());

    Ok(docker_run_cmd)
}

pub fn build_docker_run_command_with_no_gpu(
    configuration: &BuildDockerRunCommandConfiguration,
) -> Result<Vec<String>, String> {
    if configuration.docker_image_name.is_empty() {
        return Err("Docker image name is empty".to_string());
    }

    let mut docker_run_cmd = vec!["docker".to_string(), "run".to_string()];

    if configuration.is_detached {
        docker_run_cmd.push("-d".to_string());
    }
    // Remove container after exit (unless detached)
    else
    {
        docker_run_cmd.push("--rm".to_string());
    }

    // Interactive + TTY
    if configuration.is_interactive {
        docker_run_cmd.push("-it".to_string());
    }

    // Network host option (if enabled, possibly skip port mappings)
    if configuration.use_host_network {
        docker_run_cmd.push("--network".to_string());
        docker_run_cmd.push("host".to_string());
    }

    // Additional networks
    for network in &configuration.networks {
        docker_run_cmd.push("--network".to_string());
        docker_run_cmd.push(network.clone());
    }

    // Add ports from config
    for port_map in &configuration.run_config.ports {
        docker_run_cmd.push("-p".to_string());
        docker_run_cmd.push(
            format!("{}:{}",
            port_map.host_port,
            port_map.container_port));
    }

    // Volume mounts from YAML config
    for volume in &configuration.run_config.volumes {
        docker_run_cmd.push("-v".to_string());
        docker_run_cmd.push(
            format!("{}:{}",
            volume.host_path,
            volume.container_path));
    }    

    // GUI and audio support
    if configuration.enable_gui {
        add_gui_support(&mut docker_run_cmd);
    }
    if configuration.enable_audio {
        add_audio_support(&mut docker_run_cmd);
    }

    // Environment variables
    for (key, value) in &configuration.env_vars {
        docker_run_cmd.push("-e".to_string());
        docker_run_cmd.push(format!("{}={}", key, value));
    }

    // Container name
    if let Some(name) = &configuration.container_name {
        docker_run_cmd.push("--name".to_string());
        docker_run_cmd.push(name.clone());
    }

    // Entrypoint
    if let Some(entrypoint) = &configuration.entrypoint {
        docker_run_cmd.push("--entrypoint".to_string());
        docker_run_cmd.push(entrypoint.clone());
    }

    // Add image
    docker_run_cmd.push(configuration.docker_image_name.to_string());

    Ok(docker_run_cmd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configuration::run_docker_configuration::{
        PortMapping,
        VolumeMount};

    #[test]
    fn test_build_docker_run_command_with_gpu() {
        // Create a minimal configuration with GPU
        let run_config = RunDockerConfigurationData {
            volumes: vec![VolumeMount {
                host_path: "/host/data".to_string(),
                container_path: "/data".to_string(),
            }],
            ports: vec![PortMapping {
                host_port: 8080,
                container_port: 80,
            }],
        };

        let config = BuildDockerRunCommandConfiguration {
            docker_image_name: "test-image:latest".to_string(),
            run_config,
            gpu_id: Some(0),
            is_interactive: true,
            is_detached: false,
            use_host_network: true,
            container_name: Some("my-container".to_string()),
            ..Default::default()
        };

        let cmd = build_docker_run_command(&config).unwrap();

        // Uncomment the following and run `cargo test -- --nocapture` to
        //println!("Built Docker run command: {:?}", cmd);

        // Verify essential components
        assert_eq!(cmd[0], "docker");
        assert_eq!(cmd[1], "run");
        assert!(cmd.contains(&"--rm".to_string()));
        assert!(cmd.contains(&"-it".to_string()));
        assert!(cmd.contains(&"--gpus".to_string()));
        assert!(cmd.iter().any(|s| s.contains("device=0")));
        assert!(cmd.contains(&"--network".to_string()));
        assert!(cmd.contains(&"host".to_string()));
        assert!(cmd.contains(&"--name".to_string()));
        assert!(cmd.contains(&"my-container".to_string()));
        assert!(cmd.contains(&"-v".to_string()));
        assert!(cmd.iter().any(|s| s.contains("/host/data:/data")));
        assert!(cmd.contains(&"-p".to_string()));
        assert!(cmd.iter().any(|s| s.contains("8080:80")));
        assert_eq!(cmd.last().unwrap(), "test-image:latest");
    }

    #[test]
    fn test_build_docker_run_command_no_gpu() {
        // Create configuration without GPU
        let config = BuildDockerRunCommandConfiguration {
            docker_image_name: "test-no-gpu:latest".to_string(),
            run_config: RunDockerConfigurationData::default(),
            gpu_id: None,
            is_interactive: false,
            is_detached: true,
            ..Default::default()
        };

        let cmd = build_docker_run_command_with_no_gpu(&config).unwrap();

        // Uncomment the following and run `cargo test -- --nocapture` to
        //println!("Built Docker run command: {:?}", cmd);

        // Verify no GPU flags
        assert!(!cmd.contains(&"--gpus".to_string()));
        assert!(!cmd.iter().any(|s| s.contains("device=")));
        
        // Verify detached mode (no --rm, has -d, no -it)
        assert!(!cmd.contains(&"--rm".to_string()));
        assert!(cmd.contains(&"-d".to_string()));
        assert!(!cmd.contains(&"-it".to_string()));
        
        // Verify image name at end
        assert_eq!(cmd.last().unwrap(), "test-no-gpu:latest");
    }
}