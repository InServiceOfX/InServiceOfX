use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct ResponseObject {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created_at: u64,
    #[serde(default)]
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i64>,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub output: Vec<Output>,
    #[serde(default)]
    pub parallel_tool_calls: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(default)]
    pub reasoning: Reasoning,
    #[serde(default)]
    pub store: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub text: Text,
    #[serde(default)]
    pub tool_choice: String,
    #[serde(default)]
    pub tools: Vec<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    // https://platform.openai.com/docs/api-reference/responses/object#responses-object-truncation
    // "auto", "disabled" (default)
    #[serde(default)]
    pub truncation: String,
    #[serde(default)]
    pub usage: Usage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct Output {
    #[serde(rename = "type", default)]
    pub type_: String,
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub content: Vec<Content>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct Content {
    #[serde(rename = "type", default)]
    // e.g. "type": output_text
    pub type_: String,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub annotations: Vec<Value>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct Reasoning {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    // Flexible for API values like "medium" or numbers as str; parse if needed
    pub effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

//------------------------------------------------------------------------------
/// https://platform.openai.com/docs/api-reference/responses/object#responses-object-text
/// Configuration options for a text response from the model. Can be plain text
/// or structured JSON data. 
//------------------------------------------------------------------------------
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct Text {
    #[serde(default)]
    pub format: Format,
}

//------------------------------------------------------------------------------
/// https://platform.openai.com/docs/api-reference/responses/object#responses-object-text-format
/// This is typically a field within the "text" field (object).
/// An object specifying the format that the model must output.
//------------------------------------------------------------------------------
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct Format {
    #[serde(rename = "type", default)]
    pub type_: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub input_tokens_details: InputTokensDetails,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub output_tokens_details: OutputTokensDetails,
    #[serde(default)]
    pub total_tokens: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct InputTokensDetails {
    #[serde(default)]
    pub cached_tokens: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct OutputTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: u64,
}

impl ResponseObject {
    /// Parse a ResponseObject from a JSON string
    pub fn from_json(json_str: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json_str)
    }

    /// Parse a ResponseObject from a JSON Value
    pub fn from_value(value: Value) -> Result<Self, serde_json::Error> {
        serde_json::from_value(value)
    }

    /// Convert the ResponseObject to a JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_deserialize_response_object() {
        let json_data = json!({
            "id": "resp_67ccd3a9da748190baa7f1570fe91ac604becb25c45c1d41",
            "object": "response",
            "created_at": 1741476777,
            "status": "completed",
            "error": null,
            "incomplete_details": null,
            "instructions": null,
            "max_output_tokens": null,
            "model": "gpt-4o-2024-08-06",
            "output": [
                {
                    "type": "message",
                    "id": "msg_67ccd3acc8d48190a77525dc6de64b4104becb25c45c1d41",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "The image depicts a scenic landscape with a wooden boardwalk or pathway leading through lush, green grass under a blue sky with some clouds. The setting suggests a peaceful natural area, possibly a park or nature reserve. There are trees and shrubs in the background.",
                            "annotations": []
                        }
                    ]
                }
            ],
            "parallel_tool_calls": true,
            "previous_response_id": null,
            "reasoning": {
                "effort": null,
                "summary": null
            },
            "store": true,
            "temperature": 1,
            "text": {
                "format": {
                    "type": "text"
                }
            },
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1,
            "truncation": "disabled",
            "usage": {
                "input_tokens": 328,
                "input_tokens_details": {
                    "cached_tokens": 0
                },
                "output_tokens": 52,
                "output_tokens_details": {
                    "reasoning_tokens": 0
                },
                "total_tokens": 380
            },
            "user": null,
            "metadata": {}
        });

        let response: ResponseObject = serde_json::from_value(json_data).expect(
            "Failed to deserialize");
        assert_eq!(
            response.id,
            "resp_67ccd3a9da748190baa7f1570fe91ac604becb25c45c1d41");
        assert_eq!(response.status, "completed");
        assert_eq!(response.model, "gpt-4o-2024-08-06");
        assert_eq!(response.output.len(), 1);
        assert_eq!(response.output[0].type_, "message");
        assert_eq!(response.output[0].content.len(), 1);
        assert_eq!(response.output[0].content[0].type_, "output_text");
        assert!(response.error.is_none());
        assert_eq!(response.usage.input_tokens, 328u64);

        // Test partial deserialization tolerance
        // missing other fields
        let partial_json = json!({"id": "partial_id"});
        let partial_response: ResponseObject = serde_json::from_value(
            partial_json).expect("Partial deserializes without error");
        assert_eq!(partial_response.id, "partial_id");
        // defaults to empty string
        assert_eq!(partial_response.model, "");
        // defaults to 0
        assert_eq!(partial_response.created_at, 0u64);
        // defaults to empty vec
        assert!(partial_response.output.is_empty());
        // defaults to false
        assert!(!partial_response.store);
        // Option default for missing/null, otherwise 0.0
        assert_eq!(partial_response.temperature, None);
    }

    #[test]
    fn test_serialize_response_object() {
        let response = ResponseObject {
            id: "test_id".to_string(),
            object: "response".to_string(),
            created_at: 1234567890,
            status: "completed".to_string(),
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            model: "gpt-4o".to_string(),
            output: vec![],
            parallel_tool_calls: false,
            previous_response_id: None,
            reasoning: Reasoning {
                effort: None,
                summary: None,
            },
            store: false,
            temperature: Some(0.7),
            text: Text {
                format: Format {
                    type_: "text".to_string(),
                },
            },
            tool_choice: "auto".to_string(),
            tools: vec![],
            top_p: Some(1.0),
            truncation: "disabled".to_string(),
            usage: Usage {
                input_tokens: 100,
                input_tokens_details: InputTokensDetails { cached_tokens: 0 },
                output_tokens: 50,
                output_tokens_details: OutputTokensDetails {
                    reasoning_tokens: 0,
                },
                total_tokens: 150,
            },
            user: None,
            metadata: json!({}),
        };

        let serialized = serde_json::to_value(&response).expect(
            "Failed to serialize");
        let deserialized: ResponseObject = serde_json::from_value(
            serialized.clone()).expect("Failed to deserialize after serialize");
        assert_eq!(response.id, deserialized.id);
    }

}
