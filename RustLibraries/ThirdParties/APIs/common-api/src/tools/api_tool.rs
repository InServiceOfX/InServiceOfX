use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Represents a tool definition for the "tools" array in OpenAI Responses API requests (and compatible providers like Groq).
/// This enum covers key types; extend with more variants as needed.
/// Serializes to JSON objects with "type" field matching API docs.
/// For provider-specific tools (e.g., Groq's "browser_search"), use the appropriate variant or extend.
/// Usage: Collect Vec<ApiTool> and map to Vec<Value> for ModelRequestConfiguration.tools.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ApiTool {
    /// Function calling tool (universal, OpenAI/Groq/etc.).
    /// Docs: https://platform.openai.com/docs/api-reference/responses/create#responses_create-tools-function
    Function {
        function: FunctionDefinition,
    },
    /// Code interpreter tool (universal, runs Python code).
    /// Docs: https://platform.openai.com/docs/api-reference/responses/create#responses_create-tools-code_interpreter
    CodeInterpreter {
        /// The code interpreter container configuration.
        container: CodeInterpreterContainer,
    },
    /// Browser search tool (Groq-specific example; may map to OpenAI's web_search).
    /// Simple type with no additional fields in examples.
    BrowserSearch,
    // Add more variants as implemented, e.g.:
    // WebSearch { /* fields */ },
    // Mcp { server_label: String, /* etc. */ },
    // Shell { /* */ },
    // Placeholder for custom/others (serializes as {"type": "custom"})
    #[serde(rename = "custom")]
    Custom,
}

/// Definition for a function tool.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FunctionDefinition {
    /// Function name.
    pub name: String,
    /// Description of what the function does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for parameters (required for complex functions).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Map<String, Value>>,
    // Add strict: bool, etc. per docs if needed
}

/// Container config for code interpreter (always object per docs examples).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CodeInterpreterContainer {
    /// Container type (e.g., "auto", container ID, or other).
    #[serde(rename = "type")]
    pub type_: String,
    /// Optional uploaded file IDs for object containers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub files: Option<Vec<String>>,
    /// Optional memory limit in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_limit: Option<u64>,
}

/// Convenience to convert ApiTool to Value for config.tools.
impl ApiTool {
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("ApiTool serializes to Value")
    }
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_function_tool_serialize() {
        let tool = ApiTool::Function {
            function: FunctionDefinition {
                name: "get_weather".to_string(),
                description: Some("Get current weather".to_string()),
                parameters: None,
            },
        };
        let expected = json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather"
            }
        });
        assert_eq!(tool.to_value(), expected);
    }

    #[test]
    fn test_code_interpreter_serialize() {
        let tool = ApiTool::CodeInterpreter {
            container: CodeInterpreterContainer {
                type_: "auto".to_string(),
                files: None,
                memory_limit: None,
            },
        };
        let expected = json!({
            "type": "code_interpreter",
            "container": {
                "type": "auto"
            }
        });
        let value = tool.to_value();
        assert_eq!(value, expected);
    }

    #[test]
    fn test_browser_search_serialize() {
        let tool = ApiTool::BrowserSearch;
        let value = tool.to_value();
        assert_eq!(value["type"].as_str().unwrap(), "browser_search");
    }

    #[test]
    fn test_custom_tool() {
        let custom = ApiTool::Custom;
        let value = custom.to_value();
        assert_eq!(value["type"].as_str().unwrap(), "custom");
        // For full custom objects (e.g., MCP), construct as Value and add to Vec<Value> for config.tools, e.g.:
        // let mcp_tool: Value = json!({"type": "mcp", "server_label": "deepwiki"});
        // config.tools = Some(vec![mcp_tool]);
    }
}