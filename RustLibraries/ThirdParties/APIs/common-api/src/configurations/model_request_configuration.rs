use serde::{Deserialize, Serialize};
use serde_json::{self, Value};
use serde_yaml;

use std::path::Path;

/// Reasoning effort level for thinking/reasoning models
/// "low", "medium", or "high"
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

//------------------------------------------------------------------------------
/// Configuration for LLM model requests
/// Primarily based upon OpenAI's Response API:
/// https://platform.openai.com/docs/api-reference/responses/create
/// but also we attempt to also support the Chat Completion API:
/// https://platform.openai.com/docs/api-reference/chat
//------------------------------------------------------------------------------
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ModelRequestConfiguration {
    /// Model identifier (e.g., "gpt-4", "grok-4", "llama-3-70b")
    pub model: Option<String>,

    /// Function call configuration (legacy, for Chat Completions)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<Value>,

    /// Functions/tools definitions (legacy, for Chat Completions)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub functions: Option<Vec<Value>>,

    /// Instructions for the model (used in Responses API and some providers)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// Maximum completion tokens (Groq uses this name)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,

    /// Maximum tokens (OpenAI, Grok use this name)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Number of completion choices to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    /// Enable parallel tool calls
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    /// Reasoning effort level (for thinking/reasoning models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,

    /// Response format specification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,

    /// Response model for structured output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_model: Option<Value>,

    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,

    /// Enable streaming
    #[serde(default = "default_false")]
    pub stream: bool,

    /// Temperature (0.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Tool choice configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,

    /// Tools definitions (for function calling)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,

    /// User identifier for tracking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Previous response ID for conversation continuity (Responses API,
    /// xAI/Grok).
    /// https://docs.x.ai/docs/guides/chat "Chaining the conversation"
    /// https://platform.openai.com/docs/api-reference/responses
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
}

fn default_false() -> bool {
    false
}

impl ModelRequestConfiguration {
    /// Create a new empty configuration
    pub fn new() -> Self {
        Self {
            model: None,
            function_call: None,
            functions: None,
            instructions: None,
            max_completion_tokens: None,
            max_tokens: None,
            n: None,
            parallel_tool_calls: None,
            previous_response_id: None,
            reasoning_effort: None,
            response_format: None,
            response_model: None,
            stop: None,
            stream: false,
            temperature: None,
            tool_choice: None,
            tools: None,
            user: None,
        }
    }

    /// Create configuration with a model
    pub fn with_model(model: impl Into<String>) -> Self {
        Self {
            model: Some(model.into()),
            ..Self::new()
        }
    }

    /// Set the model
    pub fn set_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set max_tokens
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set max_completion_tokens (for Groq compatibility)
    pub fn with_max_completion_tokens(
        mut self,
        max_completion_tokens: u32,
    ) -> Self {
        self.max_completion_tokens = Some(max_completion_tokens);
        self
    }

    /// Set previous_response_id
    pub fn with_previous_response_id(
        mut self,
        previous_response_id: impl Into<String>
    ) -> Self {
        self.previous_response_id = Some(previous_response_id.into());
        self
    }

    /// Set streaming
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Set reasoning effort
    pub fn with_reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    /// Set tools
    pub fn with_tools(mut self, tools: Vec<Value>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Create configuration from YAML file
    pub fn from_yaml<P: AsRef<Path>>(yaml_path: P) -> Result<Self, String> {
        let path = yaml_path.as_ref();
        if !path.exists() {
            return Err(format!("YAML file not found: {}", path.display()));
        }

        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read YAML file: {}", e))?;

        let config: ModelRequestConfiguration = serde_yaml::from_str(&content)
            .map_err(|e| format!("Failed to parse YAML: {}", e))?;

        Ok(config)
    }

    /// Convert configuration to API-compatible dictionary
    /// 
    /// This handles the differences between providers:
    /// - Groq uses `max_completion_tokens` instead of `max_tokens`
    /// - Some providers may not support all fields
    pub fn to_dict(&self) -> Result<Value, serde_json::Error> {
        // Start with model (required)
        let mut dict = serde_json::json!({});

        if let Some(ref model) = self.model {
            dict["model"] = serde_json::json!(model);
        }

        // Add all non-None fields
        if let Some(ref function_call) = self.function_call {
            dict["function_call"] = function_call.clone();
        }
        if let Some(ref functions) = self.functions {
            dict["functions"] = serde_json::json!(functions);
        }
        if let Some(ref instructions) = self.instructions {
            dict["instructions"] = serde_json::json!(instructions);
        }
        if let Some(max_completion_tokens) = self.max_completion_tokens {
            dict["max_completion_tokens"] = serde_json::json!(
                max_completion_tokens);
        }
        if let Some(max_tokens) = self.max_tokens {
            dict["max_tokens"] = serde_json::json!(max_tokens);
        }
        if let Some(n) = self.n {
            dict["n"] = serde_json::json!(n);
        }
        if let Some(parallel_tool_calls) = self.parallel_tool_calls {
            dict["parallel_tool_calls"] = serde_json::json!(
                parallel_tool_calls);
        }
        if let Some(ref previous_response_id) = self.previous_response_id {
            dict["previous_response_id"] = serde_json::json!(
                previous_response_id);
        }
        if let Some(ref reasoning_effort) = self.reasoning_effort {
            dict["reasoning_effort"] = serde_json::json!(reasoning_effort);
        }
        if let Some(ref response_format) = self.response_format {
            dict["response_format"] = response_format.clone();
        }
        if let Some(ref response_model) = self.response_model {
            dict["response_model"] = response_model.clone();
        }
        if let Some(ref stop) = self.stop {
            dict["stop"] = serde_json::json!(stop);
        }
        if self.stream {
            dict["stream"] = serde_json::json!(self.stream);
        }
        if let Some(temperature) = self.temperature {
            dict["temperature"] = serde_json::json!(temperature);
        }
        if let Some(ref tool_choice) = self.tool_choice {
            dict["tool_choice"] = tool_choice.clone();
        }
        if let Some(ref tools) = self.tools {
            dict["tools"] = serde_json::json!(tools);
        }
        if let Some(ref user) = self.user {
            dict["user"] = serde_json::json!(user);
        }

        Ok(dict)
    }

    /// Convert to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Convert to pretty JSON string
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Clear all configuration (reset to defaults)
    pub fn clear(&mut self) {
        *self = Self::new();
    }
}

impl Default for ModelRequestConfiguration {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_configuration() {
        let config = ModelRequestConfiguration::new();
        assert_eq!(config.model, None);
        assert_eq!(config.stream, false);
    }

    #[test]
    fn test_with_model() {
        let config = ModelRequestConfiguration::with_model("gpt-4");
        assert_eq!(config.model, Some("gpt-4".to_string()));
    }

    #[test]
    fn test_builder_pattern() {
        let config = ModelRequestConfiguration::new()
            .set_model("grok-4")
            .with_temperature(0.7)
            .with_max_tokens(1000)
            .with_stream(true);

        assert_eq!(config.model, Some("grok-4".to_string()));
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.max_tokens, Some(1000));
        assert_eq!(config.stream, true);
    }

    #[test]
    fn test_to_dict() {
        let config = ModelRequestConfiguration::with_model("gpt-4")
            .with_temperature(0.7)
            .with_max_tokens(1000);

        let dict = config.to_dict().unwrap();
        assert_eq!(dict["model"].as_str().unwrap(), "gpt-4");
        assert_eq!(dict["temperature"].as_f64().unwrap(), 0.7);
        assert_eq!(dict["max_tokens"].as_u64().unwrap() as u32, 1000);
    }

    #[test]
    fn test_to_dict_skips_none() {
        let config = ModelRequestConfiguration::new();
        let dict = config.to_dict().unwrap();
        
        // Should not have temperature field if None
        assert!(!dict.as_object().unwrap().contains_key("temperature"));
    }

    #[test]
    fn test_reasoning_effort() {
        let config = ModelRequestConfiguration::new()
            .with_reasoning_effort(ReasoningEffort::High);

        assert_eq!(config.reasoning_effort, Some(ReasoningEffort::High));
    }

    #[test]
    fn test_manual_field_assignment_after_construction() {
        // Create a new instance (in Rust, we say "instance" or "value")
        let mut config = ModelRequestConfiguration::new();
        
        // Verify initial state
        assert_eq!(config.temperature, None);
        assert_eq!(config.max_tokens, None);
        assert_eq!(config.tools, None);
        
        // Manually set fields after construction
        config.temperature = Some(0.8);
        config.max_tokens = Some(2000);
        
        // Create some sample tools
        let tools = vec![
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather"
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform calculations"
                }
            }),
        ];
        config.tools = Some(tools.clone());
        
        // Verify the values were set correctly
        assert_eq!(config.temperature, Some(0.8));
        assert_eq!(config.max_tokens, Some(2000));
        assert_eq!(config.tools, Some(tools));
        
        // Verify they appear in the dictionary output
        let dict = config.to_dict().unwrap();
        assert_eq!(dict["temperature"].as_f64().unwrap(), 0.8);
        assert_eq!(dict["max_tokens"].as_u64().unwrap() as u32, 2000);
        assert!(dict["tools"].is_array());
        assert_eq!(dict["tools"].as_array().unwrap().len(), 2);
    }
}