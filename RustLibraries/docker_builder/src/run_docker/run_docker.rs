//! Run Docker container - main logic for loading configs and building command

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::configuration::build_docker_configuration::BuildDockerConfiguration;
use crate::configuration::run_docker_configuration::RunDockerConfiguration;
use super::build_docker_run_command::{
    BuildDockerRunCommandConfiguration,
    build_docker_run_command,
    build_docker_run_command_with_no_gpu,
};

//------------------------------------------------------------------------------
/// Arguments from CLI
//------------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct RunDockerArgs {
    pub build_dir: PathBuf,
    pub gpu_id: Option<u32>,
    pub interactive: bool,
    pub detached: bool,
    pub entrypoint: Option<String>,
    pub network_host: bool,
    pub no_gpu: bool,
    pub gui: bool,
    pub audio: bool,
}

//------------------------------------------------------------------------------
/// Load configs and build docker run command.
///
/// # Steps:
/// 1. Load build_configuration.yml from build_dir (for docker_image_name)
/// 2. Load run_configuration.yml from build_dir (for volumes/ports)
/// 3. Populate BuildDockerRunCommandConfiguration from args + configs
/// 4. Build docker run command (Vec<String>)
///
/// # Returns
/// * `Ok((Vec<String>, String))` - Command args and image name
/// * `Err(String)` - Error loading configs or building command
//------------------------------------------------------------------------------
pub fn build_run_command_from_args(
    args: &RunDockerArgs,
) -> Result<(Vec<String>, String), String> {
    // Resolve build directory
    let build_dir = args.build_dir
        .canonicalize()
        .map_err(|e| format!(
            "Invalid build directory '{}': {}",
            args.build_dir.display(), e))?;

    println!("==> Loading configurations from: {}", build_dir.display());

    // 1. Load build_configuration.yml (for docker_image_name)
    let config_file = build_dir.join("build_configuration.yml");
    if !config_file.exists() {
        return Err(format!(
            "Build configuration file not found: {}",
            config_file.display()
        ));
    }

    let build_config = BuildDockerConfiguration::load_data(Some(&config_file))?;
    let docker_image_name = build_config.docker_image_name.clone();

    println!("    Docker image: {}", docker_image_name);

    // Check if image exists
    if !check_image_exists(&docker_image_name) {
        eprintln!(
            "\n⚠ Warning: Docker image '{}' not found locally.",
            docker_image_name);
        eprintln!("  You may need to build it first:");
        eprintln!("  docker_builder build {}\n", build_dir.display());
        // Continue anyway - let Docker give the error if they proceed
    }

    // 2. Load run_configuration.yml (optional - volumes/ports)
    let run_config_file = build_dir.join("run_configuration.yml");
    let run_config_data = if run_config_file.exists() {
        println!(
            "    Loading run configuration from: {}",
            run_config_file.display());
        RunDockerConfiguration::load_data(Some(&run_config_file))?
    } else {
        println!(
            "    Warning: Run configuration file not found (using defaults)");
        Default::default()
    };

    println!("    Volumes: {}", run_config_data.volumes.len());
    println!("    Ports: {}", run_config_data.ports.len());

    // 3. Populate BuildDockerRunCommandConfiguration
    let mut docker_run_config = BuildDockerRunCommandConfiguration::default();
    docker_run_config.docker_image_name = docker_image_name.clone();
    docker_run_config.run_config = run_config_data;

    // Set fields from CLI args
    docker_run_config.is_interactive = args.interactive;
    docker_run_config.is_detached = args.detached;
    docker_run_config.use_host_network = args.network_host;
    docker_run_config.enable_gui = args.gui;
    docker_run_config.enable_audio = args.audio;

    if let Some(entrypoint) = &args.entrypoint {
        docker_run_config.entrypoint = Some(entrypoint.clone());
    }

    // Handle GPU: --no-gpu takes precedence, then --gpu N, else use all GPUs
    if args.no_gpu {
        docker_run_config.gpu_id = None;
    } else if let Some(gpu_id) = args.gpu_id {
        docker_run_config.gpu_id = Some(gpu_id);
    }
    // If neither --no-gpu nor --gpu specified, gpu_id stays None (uses all GPUs
    // in Docker)

    // 4. Build docker run command
    println!("\n==> Building docker run command...");
    let docker_cmd = if args.no_gpu {
        build_docker_run_command_with_no_gpu(&docker_run_config)?
    } else {
        build_docker_run_command(&docker_run_config)?
    };

    println!("    Command ready ({} args)", docker_cmd.len());
    
    Ok((docker_cmd, docker_image_name))
}

//------------------------------------------------------------------------------
/// Execute a docker run command.
///
/// # Arguments
/// * `cmd` - The docker run command (from build_run_command_from_args)
/// * `working_dir` - Working directory for execution (typically the build_dir)
///
/// # Returns
/// * `Ok(())` on success
/// * `Err(String)` if execution fails
//------------------------------------------------------------------------------
pub fn execute_docker_run_command(
    cmd: &[String],
    working_dir: &Path,
) -> Result<(), String> {
    println!("\n{}", "=".repeat(80));
    println!("Starting container...");
    println!("{}", "=".repeat(80));
    println!();

    // Execute the command
    let status = Command::new(&cmd[0])
        .args(&cmd[1..])
        .current_dir(working_dir)
        .status()
        .map_err(|e| format!("Failed to execute docker command: {}", e))?;

    if !status.success() {
        return Err(format!(
            "Docker run failed with exit code: {}",
            status.code().unwrap_or(-1)
        ));
    }

    println!("\n✓ Container finished successfully!");
    Ok(())
}

//------------------------------------------------------------------------------
/// Helpful utilities.
//------------------------------------------------------------------------------

/// Check if a Docker image exists locally.
pub fn check_image_exists(image_name: &str) -> bool {
    let output = Command::new("docker")
        .args(&["images", "-q", image_name])
        .output();
    
    match output {
        Ok(out) => !String::from_utf8_lossy(&out.stdout).trim().is_empty(),
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_build_run_command_with_all_configs() {
        // Create temp directory with both config files
        let temp = TempDir::new().unwrap();

        // Create build_configuration.yml
        let build_config_yaml = r#"
docker_image_name: test-image:latest
base_image: ubuntu:22.04
dockerfile_components: []
"#;
        fs::write(temp.path().join("build_configuration.yml"), build_config_yaml).unwrap();

        // Create run_configuration.yml
        let run_config_yaml = r#"
ports:
  - host_port: 8080
    container_port: 80
volumes:
  - host_path: /host/data
    container_path: /data
"#;
        fs::write(temp.path().join("run_configuration.yml"), run_config_yaml).unwrap();

        // Create args with GPU
        let args = RunDockerArgs {
            build_dir: temp.path().to_path_buf(),
            gpu_id: Some(0),
            interactive: true,
            detached: false,
            entrypoint: Some("/bin/bash".to_string()),
            network_host: true,
            no_gpu: false,
            gui: true,
            audio: false,
        };

        // Build command
        let result = build_run_command_from_args(&args);
        assert!(result.is_ok(), "Should succeed with valid configs");

        let (cmd, image_name) = result.unwrap();

        // Uncomment the following and run `cargo test -- --nocapture` to
        //println!("Built Docker run command: {:?}", cmd);

        // Verify image name extracted from build config
        assert_eq!(image_name, "test-image:latest");

        // Verify command structure
        assert_eq!(cmd[0], "docker");
        assert_eq!(cmd[1], "run");

        // Verify GPU flag from args
        assert!(cmd.contains(&"--gpus".to_string()));
        assert!(cmd.iter().any(|s| s.contains("device=0")));

        // Verify interactive flag from args
        assert!(cmd.contains(&"-it".to_string()));

        // Verify host network from args
        assert!(cmd.contains(&"--network".to_string()));
        assert!(cmd.contains(&"host".to_string()));

        // Verify entrypoint from args
        assert!(cmd.contains(&"--entrypoint".to_string()));
        assert!(cmd.contains(&"/bin/bash".to_string()));

        // Verify volume from run_configuration.yml
        assert!(cmd.contains(&"-v".to_string()));
        assert!(cmd.iter().any(|s| s.contains("/host/data:/data")));

        // Verify port from run_configuration.yml
        assert!(cmd.contains(&"-p".to_string()));
        assert!(cmd.iter().any(|s| s.contains("8080:80")));

        // Verify GUI support enabled (from args.gui = true)
        assert!(cmd.iter().any(|s| s.contains("DISPLAY")));

        // Verify image name at end
        assert_eq!(cmd.last().unwrap(), "test-image:latest");
    }

    #[test]
    fn test_build_run_command_with_no_gpu_and_missing_run_config() {
        // Create temp directory with only build_configuration.yml
        let temp = TempDir::new().unwrap();

        // Create build_configuration.yml
        let build_config_yaml = r#"
docker_image_name: no-gpu-image:v1.0
base_image: ubuntu:20.04
dockerfile_components: []
"#;
        fs::write(
            temp.path().join("build_configuration.yml"),
            build_config_yaml).unwrap();

        // Don't create run_configuration.yml (should use defaults)

        // Create args with no GPU and detached mode
        let args = RunDockerArgs {
            build_dir: temp.path().to_path_buf(),
            gpu_id: None,
            interactive: false,
            detached: true,
            entrypoint: None,
            network_host: false,
            no_gpu: true,  // Explicitly no GPU
            gui: false,
            audio: false,
        };

        // Build command
        let result = build_run_command_from_args(&args);
        assert!(
            result.is_ok(),
            "Should succeed even without run_configuration.yml");

        let (cmd, image_name) = result.unwrap();

        // Uncomment the following and run `cargo test -- --nocapture` to
        //println!("Built Docker run command: {:?}", cmd);

        // Verify image name
        assert_eq!(image_name, "no-gpu-image:v1.0");

        // Verify NO GPU flags (because args.no_gpu = true)
        assert!(!cmd.contains(&"--gpus".to_string()));
        assert!(!cmd.iter().any(|s| s.contains("device=")));

        // Verify detached mode (no --rm, has -d, no -it)
        assert!(!cmd.contains(&"--rm".to_string()));
        assert!(cmd.contains(&"-d".to_string()));
        assert!(!cmd.contains(&"-it".to_string()));

        // Verify no host network
        let network_idx = cmd.iter().position(|s| s == "--network");
        if let Some(idx) = network_idx {
            assert_ne!(cmd.get(idx + 1), Some(&"host".to_string()));
        }

        // Verify NO volumes/ports (run_configuration.yml missing, defaults to
        // empty)
        let volume_count = cmd.iter().filter(|s| *s == "-v").count();
        let port_count = cmd.iter().filter(|s| *s == "-p").count();
        assert_eq!(
            volume_count,
            0,
            "Should have no volumes from missing run config");
        assert_eq!(
            port_count,
            0,
            "Should have no ports from missing run config");

        // Verify NO GUI/audio flags
        assert!(!cmd.iter().any(|s| s.contains("DISPLAY")));
        assert!(!cmd.iter().any(|s| s.contains("pulse")));

        // Verify image at end
        assert_eq!(cmd.last().unwrap(), "no-gpu-image:v1.0");
    }
}