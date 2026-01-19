use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct VolumeMount {
    /// Path on the host machine
    pub host_path: String,
    /// Path inside the container
    pub container_path: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct PortMapping {
    /// Host port to expose.
    pub host_port: u16,
    /// Container port to map to.
    pub container_port: u16,
}

/// Data loaded from typically a run_configuration.yml file.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct RunDockerConfigurationData {
    #[serde(default)]
    pub volumes: Vec<VolumeMount>,
    #[serde(default)]
    pub ports: Vec<PortMapping>,
}

/// Builder for loading Docker run configuration from YAML files
pub struct RunDockerConfiguration;

impl RunDockerConfiguration {
    /// Default configuration file name
    pub const DEFAULT_FILE_NAME: &'static str = "run_configuration.yml";

    /// Load run configuration data from a YAML file
    /// 
    /// # Arguments
    /// * `file_path` - Optional path to the YAML configuration file.
    ///                 If `None`, uses the default file name in the current
    ///                 directory. If file doesn't exist, returns default/empty
    ///                 configuration.
    ///
    /// # Returns
    /// * `Ok(RunDockerConfigurationData)` - Successfully loaded or default
    /// * `Err(String)` - Error message if YAML parsing fails
    pub fn load_data<P: AsRef<Path>>(
        file_path: Option<P>
    ) -> Result<RunDockerConfigurationData, String> {
        // Determine the file path. "match" in this case assigns to let path.
        // Notice how it's similar to a "switch" statement, in this case
        // matching to either Some(p) or None.
        let path = match file_path {
            Some(p) => p.as_ref().to_path_buf(),
            None => {
                // Use current directory + default filename
                std::env::current_dir()
                    .map_err(|e| format!(
                        "Failed to get current directory: {}", e))?
                    .join(Self::DEFAULT_FILE_NAME)
            }
        };

        // Return default if file doesn't exist (optional config file)
        if !path.exists() {
            return Ok(RunDockerConfigurationData::default());
        }

        // Read file content
        let content = fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read configuration file: {}", e))?;

        // Parse YAML
        let data: RunDockerConfigurationData = serde_yaml::from_str(&content)
            .map_err(|e| format!("Failed to parse YAML: {}", e))?;

        Ok(data)
    }

    /// Load from a specific directory (looks for run_configuration.yml there)
    pub fn load_from_directory<P: AsRef<Path>>(
        directory: P,
    ) -> Result<RunDockerConfigurationData, String> {
        let path = directory.as_ref().join(Self::DEFAULT_FILE_NAME);
        Self::load_data(Some(path))
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
        let config_path = temp_dir.path().join("run_docker_configuration.yml");

        let yaml_content = r#"
volumes:
  - host_path: /host/data
    container_path: /container/data
ports:
  - host_port: 8080
    container_port: 80
"#;
        
        fs::write(&config_path, yaml_content).unwrap();
        
        let config = RunDockerConfiguration::load_data(
            Some(&config_path)).unwrap();

        assert_eq!(config.volumes.len(), 1);
        assert_eq!(config.volumes[0].host_path, "/host/data");
        assert_eq!(config.volumes[0].container_path, "/container/data");
        assert_eq!(config.ports.len(), 1);
        assert_eq!(config.ports[0].host_port, 8080);
        assert_eq!(config.ports[0].container_port, 80);
    }

    #[test]
    fn test_load_data_file_not_found() {
        let config = RunDockerConfiguration::load_data(
            Some("nonexistent.yml")).unwrap();
        assert!(config.volumes.is_empty());
        assert!(config.ports.is_empty());
    }

    #[test]
    fn test_default_config() {
        let config = RunDockerConfigurationData::default();
        assert!(config.volumes.is_empty());
        assert!(config.ports.is_empty());
    }
}