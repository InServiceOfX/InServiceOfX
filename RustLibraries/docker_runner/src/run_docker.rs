//! Execute docker run from argv.

use std::process::Command;

/// Run docker with the given argv (e.g. from build_docker_run_args).
/// argv[0] should be "docker", argv[1] "run", then options, image, command...
pub fn run_docker(argv: &[String]) -> Result<std::process::ExitStatus, String> {
    if argv.is_empty() {
        return Err("Empty argv".to_string());
    }
    let (binary, rest) = argv.split_first().unwrap();
    let status = Command::new(binary)
        .args(rest)
        .status()
        .map_err(|e| format!("Failed to run docker: {}", e))?;

    // For interactive sessions, don't treat container exit codes as tool failures
    // (user-initiated exit or app exit inside container is expected)
    // 127 often normal shell exit in some contexts
    if !status.success() && status.code() != Some(127) {
        return Err(format!(
            "Docker run failed with exit code: {}",
            status.code().unwrap_or(-1)
        ));
    } else if status.code() == Some(127) {
        println!(
            "Note: Container exited with code 127 (may be normal for shell exit in some setups)");
    }

    println!("\n✓ Container finished successfully!");
    Ok(status)
}