use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

const CONFIG_FILES: [(&str, &str); 6] = [
    ("batch", "batch_processing_configuration.yml"),
    ("flux", "flux_generation_configuration.yml"),
    ("nunchaku", "nunchaku_configuration.yml"),
    ("control", "nunchaku_flux_control_configuration.yml"),
    ("loras", "nunchaku_loras_configuration.yml"),
    ("pipeline", "pipeline_inputs.yml"),
];

const EDITABLE_KEYS: [&str; 4] = ["batch", "flux", "loras", "pipeline"];

#[derive(Clone)]
struct AppState {
    repo_root: PathBuf,
    config_dir: Arc<Mutex<PathBuf>>,
}

impl AppState {
    fn config_dir(&self) -> Result<PathBuf, String> {
        self.config_dir
            .lock()
            .map_err(|_| "config dir lock poisoned".to_string())
            .map(|path| path.clone())
    }

    fn set_config_dir(&self, path: PathBuf) -> Result<(), String> {
        let mut config_dir = self
            .config_dir
            .lock()
            .map_err(|_| "config dir lock poisoned".to_string())?;
        *config_dir = path;
        Ok(())
    }

    fn default_config_dir(&self) -> PathBuf {
        self.repo_root
            .join("PythonApplications")
            .join("CLIImage")
            .join("Configurations")
    }
}

#[derive(Debug)]
struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

#[derive(Debug, Serialize)]
struct ProfileSummary {
    name: String,
    complete: bool,
    missing: Vec<String>,
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    let repo_root = parse_arg_value(&args, "--repo-root")
        .map(PathBuf::from)
        .unwrap_or(env::current_dir().map_err(|e| e.to_string())?);
    let host = parse_arg_value(&args, "--host").unwrap_or_else(|| "127.0.0.1".to_string());
    let port = parse_arg_value(&args, "--port").unwrap_or_else(|| "8876".to_string());

    let config_dir = repo_root
        .join("PythonApplications")
        .join("CLIImage")
        .join("Configurations");
    let state = AppState {
        repo_root,
        config_dir: Arc::new(Mutex::new(config_dir)),
    };
    let address = format!("{}:{}", host, port);
    let listener =
        TcpListener::bind(&address).map_err(|e| format!("Failed to bind {}: {}", address, e))?;

    println!("CLIImage Config Studio backend: http://{}", address);
    println!("Config dir: {}", state.config_dir()?.display());

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let state = state.clone();
                std::thread::spawn(move || {
                    if let Err(error) = handle_connection(stream, &state) {
                        eprintln!("request error: {}", error);
                    }
                });
            }
            Err(error) => eprintln!("connection error: {}", error),
        }
    }

    Ok(())
}

fn parse_arg_value(args: &[String], key: &str) -> Option<String> {
    args.windows(2)
        .find(|window| window[0] == key)
        .map(|window| window[1].clone())
}

fn handle_connection(mut stream: TcpStream, state: &AppState) -> Result<(), String> {
    let request = read_request(&mut stream)?;
    let response = route_request(&request, state);
    stream
        .write_all(response.as_bytes())
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn read_request(stream: &mut TcpStream) -> Result<Request, String> {
    let mut buffer = Vec::new();
    let mut temp = [0_u8; 4096];
    let header_end;

    loop {
        let read_count = stream.read(&mut temp).map_err(|e| e.to_string())?;
        if read_count == 0 {
            return Err("empty request".to_string());
        }
        buffer.extend_from_slice(&temp[..read_count]);
        if let Some(index) = find_header_end(&buffer) {
            header_end = index;
            break;
        }
        if buffer.len() > 1024 * 1024 {
            return Err("request headers too large".to_string());
        }
    }

    let headers = String::from_utf8_lossy(&buffer[..header_end]);
    let mut lines = headers.lines();
    let request_line = lines
        .next()
        .ok_or_else(|| "missing request line".to_string())?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts.next().unwrap_or("").to_string();
    let path = request_parts.next().unwrap_or("").to_string();
    let mut content_length = 0_usize;
    for line in lines {
        // HTTP headers are case-insensitive (RFC 7230 §3.2).
        // Node.js (used by Vite's dev-server proxy) normalises header names to
        // lowercase, so compare case-insensitively to catch both "Content-Length:"
        // (direct curl / production) and "content-length:" (Vite proxy).
        let line_lower = line.to_ascii_lowercase();
        if let Some(value) = line_lower.strip_prefix("content-length:") {
            content_length = value.trim().parse::<usize>().unwrap_or(0);
        }
    }

    let body_start = header_end + 4;
    while buffer.len() < body_start + content_length {
        let read_count = stream.read(&mut temp).map_err(|e| e.to_string())?;
        if read_count == 0 {
            break;
        }
        buffer.extend_from_slice(&temp[..read_count]);
    }
    let body = buffer[body_start..body_start + content_length].to_vec();

    Ok(Request { method, path, body })
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

fn route_request(request: &Request, state: &AppState) -> String {
    let result = match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/api/config") => load_config(state).map(|value| json_response(200, &value)),
        ("GET", "/api/status") => status(state).map(|value| json_response(200, &value)),
        ("GET", "/api/location") => location(state).map(|value| json_response(200, &value)),
        ("POST", "/api/location") => {
            set_location(state, &request.body).map(|value| json_response(200, &value))
        }
        ("GET", "/api/profiles") => list_profiles(state)
            .map(|profiles| json_response(200, &json!({ "profiles": profiles }))),
        ("POST", "/api/config") => {
            save_config(state, &request.body).map(|_| json_response(200, &json!({ "ok": true })))
        }
        ("POST", "/api/profiles/save") => {
            save_profile(state, &request.body).map(|value| json_response(200, &value))
        }
        ("POST", "/api/profiles/apply") => {
            apply_profile(state, &request.body).map(|value| json_response(200, &value))
        }
        ("GET", "/health") => Ok(json_response(200, &json!({ "ok": true }))),
        _ => Ok(json_response(404, &json!({ "error": "not found" }))),
    };

    match result {
        Ok(response) => response,
        Err(error) => json_response(400, &json!({ "error": error })),
    }
}

fn json_response(status: u16, value: &Value) -> String {
    let body = serde_json::to_string_pretty(value).unwrap_or_else(|_| "{}".to_string());
    response(status, "application/json; charset=utf-8", &body)
}

fn response(status: u16, content_type: &str, body: &str) -> String {
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "OK",
    };
    format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Content-Type\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nConnection: close\r\n\r\n{}",
        status,
        status_text,
        content_type,
        body.as_bytes().len(),
        body
    )
}

fn load_config(state: &AppState) -> Result<Value, String> {
    let config_dir = state.config_dir()?;
    let mut map = serde_json::Map::new();
    for (key, filename) in CONFIG_FILES {
        map.insert(key.to_string(), read_yaml_json(&config_dir.join(filename))?);
    }
    map.insert(
        "profiles".to_string(),
        serde_json::to_value(list_profiles(state)?).map_err(|e| e.to_string())?,
    );
    Ok(Value::Object(map))
}

fn read_yaml_json(path: &Path) -> Result<Value, String> {
    if !path.exists() {
        return Ok(json!({}));
    }
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let yaml_value: serde_yaml::Value = serde_yaml::from_str(&text).map_err(|e| e.to_string())?;
    serde_json::to_value(yaml_value).map_err(|e| e.to_string())
}

fn write_yaml_json(path: &Path, value: &Value) -> Result<(), String> {
    let yaml_value: serde_yaml::Value =
        serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
    let text = serde_yaml::to_string(&yaml_value).map_err(|e| e.to_string())?;
    fs::write(path, text).map_err(|e| e.to_string())
}

fn save_config(state: &AppState, body: &[u8]) -> Result<(), String> {
    let config_dir = state.config_dir()?;
    let payload: Value = serde_json::from_slice(body).map_err(|e| e.to_string())?;
    for key in EDITABLE_KEYS {
        if let Some(value) = payload.get(key) {
            let filename = CONFIG_FILES
                .iter()
                .find(|(candidate, _)| candidate == &key)
                .map(|(_, filename)| filename)
                .ok_or_else(|| format!("unknown config key: {}", key))?;
            write_yaml_json(&config_dir.join(filename), value)?;
        }
    }
    Ok(())
}

fn location(state: &AppState) -> Result<Value, String> {
    Ok(json!({
        "config_dir": state.config_dir()?.to_string_lossy(),
        "default_config_dir": state.default_config_dir().to_string_lossy(),
    }))
}

#[derive(Deserialize)]
struct LocationRequest {
    config_dir: String,
    initialize_from_examples: Option<bool>,
}

fn set_location(state: &AppState, body: &[u8]) -> Result<Value, String> {
    let request: LocationRequest = serde_json::from_slice(body).map_err(|e| e.to_string())?;
    let requested_path = request.config_dir.trim();
    if requested_path.is_empty() {
        return Err("config_dir is required".to_string());
    }

    let config_dir = if Path::new(requested_path).is_absolute() {
        PathBuf::from(requested_path)
    } else {
        state.repo_root.join(requested_path)
    };

    if request.initialize_from_examples.unwrap_or(false) {
        initialize_from_examples(state, &config_dir)?;
    } else if !config_dir.exists() {
        return Err(format!(
            "config directory not found: {}",
            config_dir.display()
        ));
    }

    state.set_config_dir(config_dir)?;
    location(state)
}

fn initialize_from_examples(state: &AppState, config_dir: &Path) -> Result<(), String> {
    fs::create_dir_all(config_dir).map_err(|e| e.to_string())?;
    let example_dir = state.default_config_dir();
    for (_, filename) in CONFIG_FILES {
        let destination = config_dir.join(filename);
        if destination.exists() {
            continue;
        }
        let source = example_dir.join(format!("{}.example", filename));
        if !source.exists() {
            return Err(format!("example file not found: {}", source.display()));
        }
        fs::copy(source, destination).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn profiles_dir(state: &AppState) -> Result<PathBuf, String> {
    Ok(state.config_dir()?.join("profiles"))
}

fn profile_path(state: &AppState, name: &str) -> Result<PathBuf, String> {
    if name.is_empty() || name == "." || name == ".." || name.contains('/') || name.contains('\\') {
        return Err(format!("invalid profile name: {}", name));
    }
    if name.starts_with('_') {
        return Err("profile names starting with '_' are reserved".to_string());
    }
    Ok(profiles_dir(state)?.join(name))
}

fn list_profiles(state: &AppState) -> Result<Vec<ProfileSummary>, String> {
    let dir = profiles_dir(state)?;
    if !dir.exists() {
        return Ok(vec![]);
    }

    let mut profiles = vec![];
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('_') {
            continue;
        }
        let missing: Vec<String> = CONFIG_FILES
            .iter()
            .filter_map(|(_, filename)| {
                if path.join(filename).exists() {
                    None
                } else {
                    Some(filename.to_string())
                }
            })
            .collect();
        profiles.push(ProfileSummary {
            name,
            complete: missing.is_empty(),
            missing,
        });
    }
    profiles.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(profiles)
}

#[derive(Deserialize)]
struct ProfileRequest {
    name: String,
}

fn save_profile(state: &AppState, body: &[u8]) -> Result<Value, String> {
    let request: ProfileRequest = serde_json::from_slice(body).map_err(|e| e.to_string())?;
    let config_dir = state.config_dir()?;
    let destination = profile_path(state, request.name.trim())?;
    fs::create_dir_all(&destination).map_err(|e| e.to_string())?;
    let copied = copy_known_files(&config_dir, &destination)?;
    if copied.is_empty() {
        return Err("no live config files found".to_string());
    }
    Ok(json!({ "name": request.name, "files": copied }))
}

fn apply_profile(state: &AppState, body: &[u8]) -> Result<Value, String> {
    let request: ProfileRequest = serde_json::from_slice(body).map_err(|e| e.to_string())?;
    let config_dir = state.config_dir()?;
    let source = profile_path(state, request.name.trim())?;
    if !source.exists() {
        return Err(format!("profile not found: {}", request.name));
    }
    let backup = backup_live(state)?;
    let copied = copy_known_files(&source, &config_dir)?;
    if copied.is_empty() {
        return Err("profile has no known config files".to_string());
    }
    Ok(json!({ "name": request.name, "files": copied, "backup": backup }))
}

fn copy_known_files(source_dir: &Path, destination_dir: &Path) -> Result<Vec<String>, String> {
    fs::create_dir_all(destination_dir).map_err(|e| e.to_string())?;
    let mut copied = vec![];
    for (_, filename) in CONFIG_FILES {
        let source = source_dir.join(filename);
        if source.exists() {
            fs::copy(&source, destination_dir.join(filename)).map_err(|e| e.to_string())?;
            copied.push(filename.to_string());
        }
    }
    Ok(copied)
}

fn backup_live(state: &AppState) -> Result<Value, String> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_secs();
    let backup_dir = profiles_dir(state)?
        .join("_backups")
        .join(format!("{}", timestamp));
    let copied = copy_known_files(&state.config_dir()?, &backup_dir)?;
    if copied.is_empty() {
        Ok(Value::Null)
    } else {
        Ok(json!(backup_dir.to_string_lossy()))
    }
}

fn status(state: &AppState) -> Result<Value, String> {
    let config = load_config(state)?;
    let flux = config.get("flux").unwrap_or(&Value::Null);
    let nunchaku = config.get("nunchaku").unwrap_or(&Value::Null);
    let batch = config.get("batch").unwrap_or(&Value::Null);
    let pipeline = config.get("pipeline").unwrap_or(&Value::Null);
    let loras = config
        .get("loras")
        .and_then(|value| value.get("loras"))
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    let active_loras: Vec<Value> = loras
        .into_iter()
        .filter(|lora| {
            lora.get("is_active")
                .and_then(Value::as_bool)
                .unwrap_or(true)
        })
        .collect();
    let model_paths = nunchaku.get("nunchaku_model_paths").unwrap_or(&Value::Null);
    let model_count = if let Some(paths) = model_paths.as_array() {
        paths.len()
    } else if model_paths.is_string() {
        1
    } else {
        0
    };

    Ok(json!({
        "cuda_device": nunchaku.get("cuda_device"),
        "flux_model_path": nunchaku.get("flux_model_path"),
        "nunchaku_model_count": model_count,
        "width": flux.get("width"),
        "height": flux.get("height"),
        "steps": flux.get("num_inference_steps"),
        "guidance_scale": flux.get("guidance_scale"),
        "true_cfg_scale": flux.get("true_cfg_scale"),
        "output_path": flux.get("temporary_save_path"),
        "batch_images": batch.get("number_of_images"),
        "prompt": pipeline.get("prompt"),
        "negative_prompt": pipeline.get("negative_prompt"),
        "active_lora_count": active_loras.len(),
        "active_loras": active_loras,
    }))
}
