//! Format the docker run argv as a single shell-like string for display.

/// Format argv as a string suitable for printing (and copy-paste).
/// Args that need quoting (spaces, empty, or special chars) are wrapped in
/// double quotes with escapes.
pub fn format_command_line(args: &[String]) -> String {
    args.iter()
        .map(|a| shell_quote(a))
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_quote(s: &str) -> String {
    if s.is_empty() {
        return "\"\"".to_string();
    }
    let needs_quote = s.contains(' ') ||
        s.contains('\t') ||
        s.contains('"') ||
        s.contains('\\') ||
        s.contains('$') ||
        s.contains('`');
    if !needs_quote {
        return s.to_string();
    }
    let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{}\"", escaped)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Format the same argv as the first SGLang example (device=1, local model
    /// path, no env) and assert the full command string is as expected.
    #[test]
    fn test_format_command_line_sglang_example() {
        let argv = vec![
            "docker".to_string(),
            "run".to_string(),
            "--gpus".to_string(),
            "device=1".to_string(),
            "--shm-size".to_string(),
            "16g".to_string(),
            "-p".to_string(),
            "30000:30000".to_string(),
            "--ipc".to_string(),
            "host".to_string(),
            "-v".to_string(),
            "/media/propdev/9dc1a908-7eff-4e1c-8231-ext4/home/propdev/Data/Models/LLM/Qwen/Qwen3-4B:/models"
                .to_string(),
            "lmsysorg/sglang:latest-cu130".to_string(),
            "python3".to_string(),
            "-m".to_string(),
            "sglang.launch_server".to_string(),
            "--model-path".to_string(),
            "/models".to_string(),
            "--host".to_string(),
            "0.0.0.0".to_string(),
            "--port".to_string(),
            "30000".to_string(),
        ];
        let cmd_line = format_command_line(&argv);
        let expected = "docker run --gpus device=1 --shm-size 16g -p 30000:30000 --ipc host \
            -v /media/propdev/9dc1a908-7eff-4e1c-8231-ext4/home/propdev/Data/Models/LLM/Qwen/Qwen3-4B:/models \
            lmsysorg/sglang:latest-cu130 python3 -m sglang.launch_server --model-path /models --host 0.0.0.0 --port 30000";
        assert_eq!(
            cmd_line, expected,
            "full command line should match SGLang example");
    }
}