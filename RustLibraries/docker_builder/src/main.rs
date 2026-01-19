//! Universal Docker builder and runner

use clap::{Parser, Subcommand};
use std::path::PathBuf;

use docker_builder::run_docker::run_docker::{
    build_run_command_from_args,
    execute_docker_run_command,
    RunDockerArgs};

#[derive(Parser, Debug)]
#[command(name = "docker_builder")]
#[command(about = "Build and run Docker containers from YAML")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Build a Docker image
    Build {
        /// Path to the build directory
        build_dir: PathBuf,

        #[arg(long)]
        no_cache: bool,

        #[arg(long)]
        network_host: bool,
    },

    /// Run a Docker container
    Run {
        //----------------------------------------------------------------------
        /// Directory containing build_configuration.yml and
        /// run_configuration.yml
        //----------------------------------------------------------------------
        #[arg(long, default_value = ".")]
        build_dir: PathBuf,

        /// Specific GPU ID to use (0, 1, etc.). If not specified, uses all
        /// GPUs. (default: None)
        #[arg(long)]
        gpu_id: Option<u32>,

        /// Don't run in interactive mode (-it) (default: true)
        #[arg(long, action = clap::ArgAction::SetTrue)]
        no_interactive: bool,

        /// Run in detached mode (-d)
        #[arg(long)]
        detached: bool,

        /// Override the entrypoint (e.g., /bin/bash)
        #[arg(long)]
        entrypoint: Option<String>,

        /// Use host networking (--network host)
        #[arg(long)]
        network_host: bool,

        /// Run with no GPU
        #[arg(long)]
        no_gpu: bool,

        /// Enable GUI support (X11 forwarding)
        #[arg(long)]
        gui: bool,

        /// Enable audio support (PulseAudio + ALSA)
        #[arg(long)]
        audio: bool,
    },
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Build { build_dir, no_cache, network_host } => {
            build_docker_image(build_dir, !no_cache, network_host)
        }
        Commands::Run {
            build_dir,
            gpu_id,
            no_interactive,
            detached,
            entrypoint,
            network_host,
            no_gpu,
            gui,
            audio,
        } => {
            let interactive = !no_interactive;
            run_docker_container(
                build_dir,
                gpu_id,
                interactive,
                detached,
                entrypoint,
                network_host,
                no_gpu,
                gui,
                audio)
        }
    }
}

fn build_docker_image(
    _build_dir: PathBuf,
    _use_cache: bool,
    _use_host_network: bool,
) -> Result<(), String> {
    // ... build logic
    Ok(())
}

fn run_docker_container(
    build_dir: PathBuf,
    gpu_id: Option<u32>,
    interactive: bool,
    detached: bool,
    entrypoint: Option<String>,
    network_host: bool,
    no_gpu: bool,
    gui: bool,
    audio: bool,
) -> Result<(), String> {

    // Build args struct
    let args = RunDockerArgs {
        build_dir: build_dir.clone(),
        gpu_id: gpu_id,
        interactive: interactive,
        detached: detached,
        entrypoint,
        network_host,
        no_gpu,
        gui,
        audio,
    };

    let (docker_cmd, docker_image_name) = build_run_command_from_args(
        &args
    )?;

    // Display the command
    println!("\n==> Docker run command:");
    println!("    {}", docker_cmd.join(" "));
    println!("\n==> Image: {}", docker_image_name);

    // Execute the command
    execute_docker_run_command(&docker_cmd, &build_dir)?;

    Ok(())
}
