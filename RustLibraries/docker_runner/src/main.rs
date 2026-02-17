//! Run a Docker container from a run configuration YAML file.
//!
//! All docker run options (image, gpus, ports, volumes, env, command, etc.)
//! come from the config file. The only argument is the path to that file.

use docker_runner::build_run_command::build_docker_run_args;
use docker_runner::configuration::run_configuration::RunConfiguration;
use docker_runner::display_run_command::format_command_line;
use docker_runner::run_docker::run_docker;

fn main() {
    // Optional path to run_configuration.yml (or whatever the user calls it).
    // Default: run_configuration.yml in the current working directory.
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "run_configuration.yml".to_string());

    let config = match RunConfiguration::load_from_path(&config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error loading config from '{}': {}", config_path, e);
            std::process::exit(1);
        }
    };

    let argv = match build_docker_run_args(&config) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error building command: {}", e);
            std::process::exit(1);
        }
    };

    println!("\n==> Docker run command:\n    {}", format_command_line(&argv));

    match run_docker(&argv) {
        Ok(status) => {
            if !status.success() {
                std::process::exit(status.code().unwrap_or(1) as i32);
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}