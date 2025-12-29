//------------------------------------------------------------------------------
/// For Grok, xAI, by running this command, deliberately making the error in
/// specifying the name of the tool (e.g. "code_execution"),
///
/// curl https://api.x.ai/v1/responses \
///   -H "Content-Type: application/json" \
///   -H "Authorization: Bearer $XAI_API_KEY" \
///   -d '{
///   "model": "grok-4-1-fast",
///   "input": [
///     {
///       "role": "user",
///       "content": "Calculate the compound interest for $10,000 at 5% annually for 10 years"
///     }
///   ],
///   "tools": [
///     {
///       "type": "code_execution"
///     }
///   ]
/// }'
///
/// we find out that these are the available "built-in" tools:
/// expected one of `function`, `web_search`, `x_search`, `file_search`,
/// `code_interpreter`
//------------------------------------------------------------------------------

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Represents a tool definition for the "tools" array in OpenAI Responses API
/// requests (and compatible providers like Groq and xAI).
/// This enum covers key types; extend with more variants as needed.
/// Serializes to JSON objects with "type" field matching API docs.
/// For provider-specific tools (e.g., Groq's "browser_search"), use the
/// appropriate variant or extend.
/// Usage: Collect Vec<ApiTool> and map to Vec<Value> for
/// `ModelRequestConfiguration.tools`.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum APITools {
    /// Function calling tool (universal, OpenAI/Groq/etc.).
    /// Docs: https://platform.openai.com/docs/api-reference/responses/create#responses_create-tools-function
    Function {
        function: FunctionDefinition,
    },
    /// Code interpreter tool (Also used in Groq).
    /// Docs: https://platform.openai.com/docs/api-reference/responses/create#responses_create-tools-code_interpreter
    /// https://console.groq.com/docs/responses-api#code-execution-example
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

//------------------------------------------------------------------------------
/// https://platform.openai.com/docs/api-reference/responses/create#responses_create-tools-code_interpreter-container
/// Container config for code interpreter (always object per docs examples).
//------------------------------------------------------------------------------
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CodeInterpreterContainer {
    /// Container type (defaults to "auto" per API docs; required field, but
    /// future-proof as string).
    /// Always "auto" currently, but could be container ID or other in future.
    #[serde(rename = "type", default = "auto")]
    pub type_: String,
    /// Optional uploaded file IDs for object containers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub files_ids: Option<Vec<String>>,
    /// Optional memory limit in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_limit: Option<u64>,
}

impl Default for CodeInterpreterContainer {
    fn default() -> Self {
        Self {
            type_: "auto".to_string(),
            files_ids: None,
            memory_limit: None,
        }
    }
}

//------------------------------------------------------------------------------
/// Used by CodeInterpreterContainer for type_ field to provide default value
/// for the "type" field in the CodeInterpreterContainer.
//------------------------------------------------------------------------------
fn auto() -> String {
    "auto".to_string()
}

/// Convenience to convert APITools to Value for config.tools.
impl APITools {
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
        let tool = APITools::Function {
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
        let tool = APITools::CodeInterpreter {
            container: CodeInterpreterContainer {
                type_: "auto".to_string(),
                files_ids: None,
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
    fn test_code_interpreter_default() {
        let container = CodeInterpreterContainer::default();
        assert_eq!(container.type_, "auto");
        assert_eq!(container.files_ids, None);
        assert_eq!(container.memory_limit, None);

        // Test serde deserialization defaults missing "type" to "auto"
        let json_missing_type = json!({ "files_ids": [] });
        let deser: CodeInterpreterContainer = serde_json::from_value(
            json_missing_type).unwrap();
        assert_eq!(deser.type_, "auto");

        let json_with_type = json!(
            { "type": "custom-container", "memory_limit": 1024 });
        let deser_custom: CodeInterpreterContainer = serde_json::from_value(
            json_with_type).unwrap();
        assert_eq!(deser_custom.type_, "custom-container");
        assert_eq!(deser_custom.memory_limit, Some(1024));
    }

    #[test]
    fn test_browser_search_serialize() {
        let tool = APITools::BrowserSearch;
        let value = tool.to_value();
        assert_eq!(value["type"].as_str().unwrap(), "browser_search");
    }

    #[test]
    fn test_custom_tool() {
        let custom = APITools::Custom;
        let value = custom.to_value();
        assert_eq!(value["type"].as_str().unwrap(), "custom");
        // For full custom objects (e.g., MCP), construct as Value and add to
        // Vec<Value> for config.tools, e.g.:
        // let mcp_tool: Value =
        //     json!({"type": "mcp", "server_label": "deepwiki"});
        // config.tools = Some(vec![mcp_tool]);
    }
}