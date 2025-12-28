use serde::{Deserialize, Serialize};

/// Tool choice mode for OpenAI Responses API and compatible providers.
/// This enum represents the simple string modes for `tool_choice`.
/// For complex object configurations (e.g., `{"type": "allowed_tools", ...}`),
/// use `serde_json::Value` directly when setting in
/// `ModelRequestConfiguration`.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceMode {
    /// Model will not call any tool, generates message only.
    None,
    /// Model decides whether to generate message or call tool(s).
    Auto,
    /// Model must call one or more tools.
    Required,
}

impl std::fmt::Display for ToolChoiceMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolChoiceMode::None => write!(f, "none"),
            ToolChoiceMode::Auto => write!(f, "auto"),
            ToolChoiceMode::Required => write!(f, "required"),
        }
    }
}

impl ToolChoiceMode {
    /// Convert to JSON value (string) for use in requests.
    pub fn to_value(&self) -> serde_json::Value {
        serde_json::Value::String(self.to_string())
    }
}

impl std::str::FromStr for ToolChoiceMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_lowercase();
        match lower.as_str() {
            "none" => Ok(Self::None),
            "auto" => Ok(Self::Auto),
            "required" => Ok(Self::Required),
            _ => Err(format!("Unknown ToolChoiceMode: {}", s)),
        }
    }
}

// Unit tests
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use crate::configurations::model_request_configuration::ModelRequestConfiguration;

    #[test]
    fn test_serialize() {
        assert_eq!(
            serde_json::to_value(ToolChoiceMode::Auto).unwrap(),
            json!("auto")
        );
        assert_eq!(
            serde_json::to_value(ToolChoiceMode::None).unwrap(),
            json!("none")
        );
        assert_eq!(
            serde_json::to_value(ToolChoiceMode::Required).unwrap(),
            json!("required")
        );
    }

    #[test]
    fn test_deserialize() {
        assert_eq!(
            "auto".parse::<ToolChoiceMode>().unwrap(),
            ToolChoiceMode::Auto);
        // Note: deserialize from string works via serde rename
        let v: ToolChoiceMode = serde_json::from_value(json!("none")).unwrap();
        assert_eq!(v, ToolChoiceMode::None);
    }

    #[test]
    fn test_to_value() {
        assert_eq!(ToolChoiceMode::Required.to_value(), json!("required"));
    }

    #[test]
    fn test_works_with_model_request_configuration() {
        let mut configuration = ModelRequestConfiguration::new();
        assert_eq!(configuration.tool_choice, None);
        configuration.tool_choice = Some(ToolChoiceMode::Required.to_value());
        assert_eq!(
            configuration.tool_choice,
            Some(ToolChoiceMode::Required.to_value()));

        let dict = configuration.to_dict().unwrap();
        assert!(dict.as_object().unwrap().contains_key("tool_choice"));
        assert_eq!(dict["tool_choice"].as_str().unwrap(), "required");
    }
}
