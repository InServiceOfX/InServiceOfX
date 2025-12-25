
use serde::{Deserialize, Serialize};
use serde_json::{self, Value};
use sha2::{Digest, Sha256};
use std::fmt;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Developer,
}

/// A message in the chat completion request
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    /// Create a new message
    pub fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Self { role, content: content.into(), name: None, }
    }

    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
            name: None,
        }
    }

    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
            name: None,
        }
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
            name: None,
        }
    }

    /// Create a developer message
    /// Equivalent to Python's DeveloperMessage
    pub fn developer(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Developer,
            content: content.into(),
            name: None,
        }
    }

    /// Set the name field (for system/user/developer messages with names)
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Generate SHA256 hash of message content
    /// Equivalent to Python's `_hash_content()` static method
    pub fn hash_content(&self) -> String {
        Self::hash_content_static(&self.content)
    }

    /// Static method to hash content string
    pub fn hash_content_static(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Convert message to string with optional prefix override and return
    /// character and word counts of the content
    pub fn to_string_and_counts(&self, prefix: Option<&str>)
        -> (String, usize, usize)
    {
        let role_str = prefix.unwrap_or(match self.role {
            MessageRole::System => "System",
            MessageRole::User => "Human",
            MessageRole::Assistant => "AI",
            MessageRole::Developer => "Developer",
        });

        let formatted = format!("{}: {}", role_str, self.content);
        let char_count = self.content.chars().count();
        let word_count = self.content.split_whitespace().count();

        (formatted, char_count, word_count)
    }

    /// Convert to dictionary format for API requests
    pub fn to_dict(&self) -> Result<Value, serde_json::Error> {
        serde_json::to_value(self)
    }

    /// Convert to JSON string
    /// Useful for debugging or direct API calls
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Convert to pretty JSON string (for debugging/display)
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

impl fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (formatted, _, _) = self.to_string_and_counts(None);
        write!(f, "{}", formatted)
    }
}

pub fn create_system_message(
    content: impl Into<String>,
    name: Option<impl Into<String>>,
) -> Message {
    let mut message = Message::system(content.into());
    if let Some(name) = name {
        message = message.with_name(name);
    }
    message
}

pub fn create_user_message(
    content: impl Into<String>,
    name: Option<impl Into<String>>,
) -> Message {
    let mut message = Message::user(content);
    if let Some(name) = name {
        message = message.with_name(name);
    }
    message
}

pub fn create_assistant_message(content: impl Into<String>) -> Message {
    Message::assistant(content.into())
}

pub fn create_developer_message(
    content: impl Into<String>,
    name: Option<impl Into<String>>,
) -> Message {
    let mut message = Message::developer(content.into());
    if let Some(name) = name {
        message = message.with_name(name.into());
    }
    message
}

pub fn parse_dict_into_message(dict: Value)
    -> Result<Message, Box<dyn std::error::Error + Send + Sync>>
{
    // Validate required fields first
    let _role_str = dict["role"].as_str().ok_or(
        "Message must have a 'role' field")?;

    let _content_str = dict["content"].as_str().ok_or(
        "Message must have a 'content' field")?;

    // Could map role_str to enum, but serde handles it automatically
    let message: Message = serde_json::from_value(dict)?;

    if message.content.is_empty() {
        return Err("Message content cannot be empty".into());
    }

    Ok(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn test_constructors() {
        let sys = Message::system("System prompt".to_string());
        assert_eq!(sys.role, MessageRole::System);
        assert_eq!(sys.content, "System prompt");
        assert_eq!(sys.name, None);

        let dev = Message::developer("Dev instructions".to_string()).with_name(
            "dev_name");
        assert_eq!(dev.role, MessageRole::Developer);
        assert_eq!(dev.content, "Dev instructions");
        assert_eq!(dev.name, Some("dev_name".to_string()));
    }

    #[test]
    fn test_serialization() {
        let msg = Message::user("User query".to_string());
        let dict = msg.to_dict().unwrap();
        assert_eq!(dict["role"], Value::String("user".to_string()));
        assert_eq!(dict["content"], Value::String("User query".to_string()));
        // Optional not set
        assert_eq!(dict["name"], Value::Null);

        let json = msg.to_json().unwrap();
        let parsed: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, msg);
    }

    #[test]
    fn test_hash_content() {
        let hash = Message::hash_content_static("test content");
        assert_eq!(hash.len(), 64);
        assert_eq!(Message::hash_content_static("test content"), hash);
        assert_ne!(Message::hash_content_static("different"), hash);
    }

    #[test]
    fn test_to_string_and_counts() {
        let msg = Message::user(
            "Hello world with 4 words and unicode cafés".to_string());
        let (s, c, w) = msg.to_string_and_counts(None);
        assert!(s.starts_with(
            "Human: Hello world with 4 words and unicode cafés"));
        // Uses chars().count() for unicode-safe char counting (matches Python
        // len(str))
        // chars().count() == 42 (unicode chars, e.g., 'é' counts as 1)
        assert_eq!(c, 42);
        assert_eq!(w, 8);

        let (custom, _, _) = msg.to_string_and_counts(Some("Custom"));
        assert!(custom.starts_with(
            "Custom: Hello world with 4 words and unicode cafés"));
    }

    #[test]
    fn test_display() {
        let msg = Message::assistant("Display test".to_string());
        let display_str = format!("{}", msg);
        assert!(display_str.starts_with("AI: Display test"));
    }

    #[test]
    fn test_create_system_message() {
        let msg = create_system_message(
            "You are a helpful AI assistant specialized in Rust programming and API development.",
            Some("RustAPIAssistant")
        );
        assert_eq!(msg.role, MessageRole::System);
        assert_eq!(msg.content, "You are a helpful AI assistant specialized in Rust programming and API development.");
        assert_eq!(msg.name, Some("RustAPIAssistant".to_string()));
        // Real-life use: serialize to JSON for API request
        let json = msg.to_json().expect("Should serialize");
        assert!(json.contains("\"role\":\"system\""));
        assert!(json.contains("\"name\":\"RustAPIAssistant\""));
    }

    #[test]
    fn test_create_user_message() {
        let msg = create_user_message(
            "How do I create a simple HTTP server in Rust that handles GET and POST requests?",
            None::<String>
        );
        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(
            msg.content,
            "How do I create a simple HTTP server in Rust that handles GET and POST requests?");
        assert!(msg.name.is_none());
        // Real-life: hash for caching or uniqueness
        let hash = msg.hash_content();
        assert_eq!(hash.len(), 64);
        assert!(!hash.is_empty());
    }

    #[test]
    fn test_create_assistant_message() {
        let msg = create_assistant_message(
            "You can use the Actix-web crate. Add it to your Cargo.toml and define routes with #[get(\"/\")] etc."
        );
        assert_eq!(msg.role, MessageRole::Assistant);
        assert_eq!(
            msg.content,
            "You can use the Actix-web crate. Add it to your Cargo.toml and define routes with #[get(\"/\")] etc.");
        // Real-life: display for logging conversation
        let display = format!("{}", msg);
        assert!(display.starts_with("AI: You can use the Actix-web crate."));
    }

    #[test]
    fn test_create_developer_message() {
        let msg = create_developer_message(
            "Implement error handling for all endpoints and log requests using tracing crate.",
            Some("Development Team")
        );
        assert_eq!(msg.role, MessageRole::Developer);
        assert_eq!(
            msg.content,
            "Implement error handling for all endpoints and log requests using tracing crate.");
        assert_eq!(msg.name, Some("Development Team".to_string()));
        // Real-life: get counts for token estimation
        let (_, char_count, word_count) = msg.to_string_and_counts(Some("Dev"));
        assert!(char_count >= 80);
        assert!(word_count > 10);
    }
}
