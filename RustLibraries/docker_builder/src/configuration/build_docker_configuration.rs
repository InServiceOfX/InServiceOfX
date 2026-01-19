use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct DockerfileComponent {
    /// Human-readable label or filename for identification (e.g.,
    /// "Dockerfile.header").
    pub label: String,
    /// Relative or absolute path to the Dockerfile fragment file.
    /// Relative paths are joined with the config directory.
    pub path: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BuildDockerConfigurationData {
    pub docker_image_name: String,
    
    /// Base Docker image to use
    pub base_image: String,
    
    /// Build arguments as key-value pairs
    /// These are typically environment variables passed to the Docker build
    /// command.
    #[serde(
        default,
        skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub build_args: std::collections::HashMap<String, String>,

    /// Ordered list of Dockerfile components to concatenate into the final
    /// Dockerfile.
    /// Each has a label (for logging/ID) and path (relative to config dir or
    /// absolute).
    #[serde(default)]
    pub dockerfile_components: Vec<DockerfileComponent>,
}

impl Default for BuildDockerConfigurationData {
    fn default() -> Self {
        Self {
            docker_image_name: String::new(),
            base_image: String::new(),
            build_args: std::collections::HashMap::new(),
            dockerfile_components: Vec::new(),
        }
    }
}

/// Builder for loading Docker build configuration from YAML files
pub struct BuildDockerConfiguration;

impl BuildDockerConfiguration {
    fn resolve_path(base: &Path, c: &DockerfileComponent) -> PathBuf {
        let p = Path::new(&c.path);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            base.join(&c.path)
        }
    }

    /// Default configuration file name
    pub const DEFAULT_FILE_NAME: &'static str = "build_configuration.yml";

    //--------------------------------------------------------------------------
    /// Load build configuration data from a YAML file
    /// 
    /// # Arguments
    /// * `file_path` - Optional path to the YAML configuration file.
    ///                  If `None`, uses the default file name in the current
    ///                  directory.
    /// 
    /// # Returns
    /// * `Ok(BuildDockerConfigurationData)` - Successfully loaded configuration
    /// * `Err(String)` - Error message if file not found, invalid YAML, or
    /// missing required fields
    /// 
    /// # Errors
    /// * Returns error if the file doesn't exist
    /// * Returns error if YAML parsing fails
    /// * Returns error if required fields (`docker_image_name`, `base_image`)
    ///   are missing
    //--------------------------------------------------------------------------
    pub fn load_data<P: AsRef<Path>>(
        file_path: Option<P>
    ) -> Result<BuildDockerConfigurationData, String> {
        // Determine the file path
        let path = match file_path {
            Some(p) => p.as_ref().to_path_buf(),
            None => {
                // Use current directory + default filename
                // Alternative: use env!("CARGO_MANIFEST_DIR") if you want
                // crate-relative path
                std::env::current_dir()
                    .map_err(|e| format!(
                        "Failed to get current directory: {}", e))?
                    .join(Self::DEFAULT_FILE_NAME)
            }
        };

        // Check if file exists
        if !path.exists() {
            return Err(format!(
                "Configuration file '{}' does not exist.",
                path.display()
            ));
        }

        // Read file content
        let content = fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read configuration file: {}", e))?;

        // Parse YAML
        let mut data: BuildDockerConfigurationData = serde_yaml::from_str(
            &content)
            .map_err(|e| format!("Failed to parse YAML: {}", e))?;

        // Validate required fields
        if data.docker_image_name.is_empty() {
            return Err(
                "Missing required field: 'docker_image_name'".to_string());
        }
        if data.base_image.is_empty() {
            return Err(
                "Missing required field: 'base_image'".to_string());
        }

        // Ensure build_args is initialized (serde default should handle this,
        // but be explicit)
        if data.build_args.is_empty() {
            data.build_args = std::collections::HashMap::new();
        }

        // Resolve Dockerfile component paths to absolute and check existence
        let config_dir = path.parent().unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        for component in &mut data.dockerfile_components {
            let resolved = Self::resolve_path(&config_dir, component);
            component.path = resolved.to_string_lossy().into_owned();
            if !resolved.exists() {
                eprintln!(
                    "Warning: Dockerfile component '{}' path does not exist: {}",
                    component.label,
                    resolved.display());
            }
        }

        Ok(data)
    }

    //--------------------------------------------------------------------------
    /// Load configuration from a default path relative to the current file
    /// 
    /// This uses `file!()` macro to get the current file's path and looks for
    /// the default configuration file in the same directory.
    //--------------------------------------------------------------------------
    pub fn load_data_from_default_path() -> Result<
        BuildDockerConfigurationData,
        String>
    {
        // Get the directory of the current source file
        let current_file = Path::new(file!());
        let default_dir = current_file.parent()
            .ok_or_else(|| "Could not determine parent directory".to_string())?;

        let default_path = default_dir.join(Self::DEFAULT_FILE_NAME);
        Self::load_data(Some(default_path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_load_data_with_valid_yaml() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("build_configuration.yml");
        
        let yaml_content = r#"
docker_image_name: test-image:latest
base_image: ubuntu:22.04
build_args:
  ARG1: value1
  ARG2: value2
"#;
        
        fs::write(&config_path, yaml_content).unwrap();
        
        let config = BuildDockerConfiguration::load_data(
            Some(&config_path)).unwrap();

        assert_eq!(config.docker_image_name, "test-image:latest");
        assert_eq!(config.base_image, "ubuntu:22.04");
        assert_eq!(config.build_args.len(), 2);
        assert_eq!(config.build_args.get("ARG1"), Some(&"value1".to_string()));
        assert_eq!(config.build_args.get("ARG2"), Some(&"value2".to_string()));
    }

    #[test]
    fn test_load_data_with_missing_required_field() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("build_configuration.yml");
        
        let yaml_content = r#"
base_image: ubuntu:22.04
"#;
        
        fs::write(&config_path, yaml_content).unwrap();
        
        let result = BuildDockerConfiguration::load_data(Some(&config_path));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("docker_image_name"));
    }

    #[test]
    fn test_load_data_with_empty_build_args() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("build_configuration.yml");
        
        let yaml_content = r#"
docker_image_name: test-image:latest
base_image: ubuntu:22.04
"#;
        
        fs::write(&config_path, yaml_content).unwrap();
        
        let config = BuildDockerConfiguration::load_data(
            Some(&config_path)).unwrap();
        assert!(config.build_args.is_empty());
    }

    #[test]
    fn test_load_data_file_not_found() {
        let result = BuildDockerConfiguration::load_data(
            Some("nonexistent.yml"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("does not exist"));
    }
}