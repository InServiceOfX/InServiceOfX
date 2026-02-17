//! Build the docker run argv from RunConfiguration.
//! Only options present in config are added; order is fixed for readability.

use crate::configuration::run_configuration::{expand_tilde, RunConfiguration};

/// Build argv for docker run:
/// ["docker", "run", ...options..., image, ...command...].
/// Volumes: host path is tilde-expanded.
pub fn build_docker_run_args(
    configuration: &RunConfiguration,
) -> Result<Vec<String>, String> {
    if configuration.docker_image_name.trim().is_empty() {
        return Err("Configuration 'docker_image_name' is empty".to_string());
    }

    let mut args = vec!["docker".to_string(), "run".to_string()];

    if let Some(ref g) = configuration.gpus {
        if !g.is_empty() {
            args.push("--gpus".to_string());
            args.push(g.clone());
        }
    }

    if let Some(ref s) = configuration.shm_size {
        if !s.is_empty() {
            args.push("--shm-size".to_string());
            args.push(s.clone());
        }
    }

    if let Some(ref port_list) = configuration.ports {
        for port_map in port_list {
            args.push("-p".to_string());
            args.push(
                format!(
                    "{}:{}", port_map.host_port, port_map.container_port));
        }
    }

    if let Some(ref vol_list) = configuration.volumes {
        for volume in vol_list {
            let host_exp = expand_tilde(volume.host_path.trim());
            args.push("-v".to_string());
            args.push(format!("{}:{}", host_exp, volume.container_path.trim()));
        }
    }

    if let Some(ref e) = configuration.env {
        for (k, v) in e.clone().into_env_pairs() {
            if !k.is_empty() {
                args.push("-e".to_string());
                args.push(format!("{}={}", k, v));
            }
        }
    }

    if let Some(ref i) = configuration.ipc {
        if !i.is_empty() {
            args.push("--ipc".to_string());
            args.push(i.clone());
        }
    }

    args.push(configuration.docker_image_name.trim().to_string());

    if let Some(ref cmd) = configuration.command {
        let parts = cmd.clone().into_vec();
        for p in parts {
            if !p.is_empty() {
                args.push(p);
            }
        }
    }

    Ok(args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configuration::run_configuration::{
        CommandOption, EnvOption, PortMapping, RunConfiguration, VolumeMount,
    };
    use std::collections::HashMap;

    /// Nominal test: build_docker_run_args produces correct argv for a full
    /// config.
    #[test]
    fn test_build_docker_run_args_full_config() {
        let mut env_map = HashMap::new();
        env_map.insert("HF_TOKEN".to_string(), "secret".to_string());
        let config = RunConfiguration {
            docker_image_name: "lmsysorg/sglang:latest-cu130".to_string(),
            gpus: Some("device=1".to_string()),
            shm_size: Some("16g".to_string()),
            ports: Some(vec![PortMapping {
                host_port: 30000,
                container_port: 30000,
            }]),
            volumes: Some(vec![
                VolumeMount {
                    host_path: "/host/models".to_string(),
                    container_path: "/models".to_string(),
                },
            ]),
            env: Some(EnvOption::Map(env_map)),
            ipc: Some("host".to_string()),
            command: Some(CommandOption::List(vec![
                "python3".to_string(),
                "-m".to_string(),
                "sglang.launch_server".to_string(),
                "--model-path".to_string(),
                "/models".to_string(),
                "--port".to_string(),
                "30000".to_string(),
            ])),
        };

        let args = build_docker_run_args(&config).expect(
            "build should succeed");
        assert!(args.len() >= 2);
        assert_eq!(args[0], "docker");
        assert_eq!(args[1], "run");
        assert!(args.contains(&"--gpus".to_string()));
        assert!(args.contains(&"device=1".to_string()));
        assert!(args.contains(&"--shm-size".to_string()));
        assert!(args.contains(&"16g".to_string()));
        assert!(args.contains(&"-p".to_string()));
        assert!(args.contains(&"30000:30000".to_string()));
        assert!(args.contains(&"-v".to_string()));
        assert!(args.iter().any(|a| a == "/host/models:/models"));
        assert!(args.contains(&"-e".to_string()));
        assert!(args.iter().any(|a| a.starts_with("HF_TOKEN=")));
        assert!(args.contains(&"--ipc".to_string()));
        assert!(args.contains(&"host".to_string()));
        assert!(args.contains(&"lmsysorg/sglang:latest-cu130".to_string()));
        assert!(args.contains(&"python3".to_string()));
        assert!(args.contains(&"sglang.launch_server".to_string()));
        assert!(args.contains(&"--model-path".to_string()));
        assert!(args.contains(&"/models".to_string()));
        assert!(args.contains(&"30000".to_string()));
    }
}