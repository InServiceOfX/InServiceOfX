use std::path::{Path, PathBuf};
use std::process::Command;

use crate::configuration::build_docker_configuration::BuildDockerConfiguration;
use super::create_dockerfile::create_dockerfile;
use super::build_docker_command::build_docker_build_command;

/// Arguments from CLI for building
#[derive(Debug, Clone)]
pub struct BuildDockerArgs {
    pub build_dir: PathBuf,
    pub no_cache: bool,
    pub network_host: bool,
}

//------------------------------------------------------------------------------
/// # Steps:
/// 1. Load build_configuration.yml from build_dir
/// 2. Create Dockerfile from components (paths already resolved in load_data)
/// 3. Build docker build command
/// 4. Execute docker build
///
/// # Returns
/// * `Ok(String)` - Built image name
/// * `Err(String)` - Error at any step
//------------------------------------------------------------------------------
pub fn build_docker_image_from_args(
    args: &BuildDockerArgs,
) -> Result<String, String> {
    // Resolve build directory
    let build_dir = args.build_dir
        .canonicalize()
        .map_err(|e| format!(
            "Invalid build directory '{}': {}",
            args.build_dir.display(), e))?;

    println!("==> Building Docker image from: {}", build_dir.display());

    // 1. Load build_configuration.yml
    let config_file = build_dir.join("build_configuration.yml");
    if !config_file.exists() {
        return Err(format!(
            "Build configuration file not found: {}",
            config_file.display()
        ));
    }

    println!("    Loading configuration from: {}", config_file.display());
    let config = BuildDockerConfiguration::load_data(Some(&config_file))?;

    println!("    Image name: {}", config.docker_image_name);
    println!("    Base image: {}", config.base_image);
    println!(
        "    Dockerfile components: {}",
        config.dockerfile_components.len());

    // 2. Create Dockerfile from components
    // Paths in dockerfile_components are already resolved to absolute paths by
    // load_data's resolve_path logic
    let dockerfile_path = build_dir.join("Dockerfile");
    println!("\n==> Creating Dockerfile at: {}", dockerfile_path.display());

    create_dockerfile(&config_file, &dockerfile_path)?;
    
    // Verify Dockerfile was created
    if !dockerfile_path.exists() {
        return Err(format!(
            "Dockerfile was not created at '{}'",
            dockerfile_path.display()
        ));
    }
    println!("    ✓ Dockerfile created successfully");

    // 3. Build docker build command
    println!("\n==> Building docker build command...");
    let docker_cmd = build_docker_build_command(
        &dockerfile_path,
        &config,
        // use_cache = !no_cache
        !args.no_cache,
        args.network_host,
    );

    println!("    Command ready ({} args)", docker_cmd.len());

    // Display the command
    println!("\n==> Docker build command:");
    println!("    {}", docker_cmd.join(" "));
    println!();

    // 4. Execute docker build
    execute_docker_build(&docker_cmd, &build_dir)?;

    println!("\n✓ Docker image built successfully!");
    println!("  Image: {}", config.docker_image_name);

    Ok(config.docker_image_name)
}

/// Execute docker build command
fn execute_docker_build(
    cmd: &[String],
    build_context: &Path,
) -> Result<(), String> {
    println!("{}", "=".repeat(80));
    println!("Building Docker image...");
    println!("{}", "=".repeat(80));
    println!();

    let status = Command::new(&cmd[0])
        .args(&cmd[1..])
        .env("DOCKER_BUILDKIT", "1")
        .current_dir(build_context)
        .status()
        .map_err(|e| format!("Failed to execute docker build: {}", e))?;

    if !status.success() {
        return Err(format!(
            "Docker build failed with exit code: {}",
            status.code().unwrap_or(-1)
        ));
    }

    Ok(())
}