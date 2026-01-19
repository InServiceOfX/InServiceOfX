use std::collections::HashMap;
use std::path::Path;

use crate::configuration::build_docker_configuration::BuildDockerConfigurationData;

/// Builds the Docker build command as a list of arguments.
///
/// # Arguments
/// * `dockerfile_path` - Path to the Dockerfile
/// * `build_configuration` - Populated BuildDockerConfigurationData
/// * `use_cache` - Whether to use Docker build cache
/// * `use_host_network` - Whether to use --network host
///
/// # Returns
/// * Vec<String> representing the full command (e.g., for exec or logging)
///
/// Note: Build context (.) and cwd are handled separately at runtime if needed.
pub fn build_docker_build_command(
    dockerfile_path: &Path,
    build_configuration: &BuildDockerConfigurationData,
    use_cache: bool,
    use_host_network: bool,
) -> Vec<String> {
    let mut docker_build_cmd = vec![
        "DOCKER_BUILDKIT=1".to_string(),
        "docker".to_string(),
        "build".to_string(),
    ];

    if !use_cache {
        docker_build_cmd.push("--no-cache".to_string());
    }

    if use_host_network {
        docker_build_cmd.push("--network".to_string());
        docker_build_cmd.push("host".to_string());
    }

    // Add build_args from configuration (dynamic from YAML)
    let build_args: &HashMap<String, String> = &build_configuration.build_args;
    for (key, value) in build_args {
        docker_build_cmd.push("--build-arg".to_string());
        docker_build_cmd.push(format!("{}={}", key.to_uppercase(), value));
    }

    // Always add base_image and docker_image_name as --build-arg
    if !build_configuration.base_image.is_empty() {
        docker_build_cmd.push("--build-arg".to_string());
        docker_build_cmd.push(format!("BASE_IMAGE={}", build_configuration.base_image));
    } else {
        // In real use, this could be logged/handled upstream
        eprintln!("Warning: base_image is empty in configuration");
    }

    if !build_configuration.docker_image_name.is_empty() {
        docker_build_cmd.push("--build-arg".to_string());
        docker_build_cmd.push(format!("DOCKER_IMAGE_NAME={}", build_configuration.docker_image_name));
    } else {
        eprintln!("Warning: docker_image_name is empty in configuration");
    }

    // Specify Dockerfile
    docker_build_cmd.push("-f".to_string());
    docker_build_cmd.push(dockerfile_path.to_string_lossy().into_owned());

    // Tag the image
    docker_build_cmd.push("-t".to_string());
    docker_build_cmd.push(build_configuration.docker_image_name.clone());

    // Build context (.)
    docker_build_cmd.push(".".to_string());

    docker_build_cmd
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_build_docker_build_command() {
        let mut build_args = HashMap::new();
        build_args.insert("ARG1".to_string(), "value1".to_string());

        let config = BuildDockerConfigurationData {
            docker_image_name: "test-image:latest".to_string(),
            base_image: "ubuntu:22.04".to_string(),
            build_args,
            dockerfile_components: vec![],
        };

        let dockerfile_path = Path::new("Dockerfile");
        let cmd = build_docker_build_command(
            dockerfile_path,
            &config,
            false,
            true,
        );

        // Uncomment the following and run `cargo test -- --nocapture` to
        // inspect the full command during tests:
        // println!("Built Docker command: {:?}", cmd);

        assert!(cmd.contains(&"--no-cache".to_string()));
        assert!(cmd.contains(&"--network".to_string()));
        assert!(cmd.contains(&"host".to_string()));
        assert!(cmd.contains(&"--build-arg".to_string()));
        assert!(cmd.iter().any(|s| s.contains("ARG1=value1")));
        assert!(cmd.iter().any(|s| s.contains("BASE_IMAGE=ubuntu:22.04")));
        assert!(cmd.iter().any(|s| s.contains(
            "DOCKER_IMAGE_NAME=test-image:latest")));
        assert!(cmd.last() == Some(&".".to_string()));
    }
}
