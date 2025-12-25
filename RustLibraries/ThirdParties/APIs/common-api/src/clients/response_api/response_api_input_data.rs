use crate::configurations::ModelRequestConfiguration;
use crate::messages::basic_messages::Message;
use serde::{Deserialize, Serialize};
use serde_json::{self, Value};

/// Stores configuration and messages for Responses API requests
/// Single responsibility: Build the JSON data for curl -d flag
/// 
/// This struct persists the configuration (which doesn't change much)
/// and maintains a list of messages to send in the request.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ResponsesAPIInputData {
    pub configuration: ModelRequestConfiguration,
    
    /// List of messages to send in the request
    pub messages: Vec<Message>,
}

impl ResponsesAPIInputData {
    /// Create a new instance with a configuration and messages
    pub fn new(
        configuration: ModelRequestConfiguration,
        messages: Vec<Message>) -> Self
    {
        Self {
            configuration,
            messages,
        }
    }

    /// Create with default empty configuration
    pub fn with_default_config() -> Self {
        Self {
            configuration: ModelRequestConfiguration::new(),
            messages: Vec::new(),
        }
    }

    /// Get a reference to the configuration
    pub fn configuration(&self) -> &ModelRequestConfiguration {
        &self.configuration
    }

    /// Get a mutable reference to the configuration
    pub fn configuration_mut(&mut self) -> &mut ModelRequestConfiguration {
        &mut self.configuration
    }

    /// Get a reference to the messages
    pub fn messages(&self) -> &Vec<Message> {
        &self.messages
    }

    /// Get a mutable reference to the messages
    pub fn messages_mut(&mut self) -> &mut Vec<Message> {
        &mut self.messages
    }

    /// Add a message to the list
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Add multiple messages
    pub fn add_messages(&mut self, messages: Vec<Message>) {
        self.messages.extend(messages);
    }

    /// Clear all messages
    pub fn clear_messages(&mut self) {
        self.messages.clear();
    }

    /// Set the previous_response_id in configuration
    pub fn set_previous_response_id(
        &mut self,
        previous_response_id: impl Into<String>,
    ) -> ()
    {
        self.configuration.previous_response_id = Some(
            previous_response_id.into());
    }

    /// Clear the previous_response_id
    pub fn clear_previous_response_id(&mut self) {
        self.configuration.previous_response_id = None;
    }

    /// Build the JSON string for curl -d flag
    /// 
    /// This combines:
    /// - All non-None values from configuration.to_dict()
    /// - The "input" field containing messages as JSON array
    /// 
    /// Returns the JSON string that goes after `-d` in curl command
    pub fn build_curl_data(&self) -> Result<String, serde_json::Error> {
        // Start with configuration dict (contains all non-None fields)
        let mut request_body = self.configuration.to_dict()?;

        // Convert messages to JSON array
        let messages_json: Vec<Value> = self.messages
            .iter()
            .map(|msg| msg.to_dict())
            .collect::<Result<Vec<Value>, serde_json::Error>>()?;

        // Add input field with messages array
        request_body["input"] = serde_json::json!(messages_json);

        // Serialize to JSON string
        serde_json::to_string(&request_body)
    }

    /// Build the JSON string in pretty format (for debugging/display)
    pub fn build_curl_data_pretty(&self) -> Result<String, serde_json::Error> {
        let mut request_body = self.configuration.to_dict()?;
        
        let messages_json: Vec<Value> = self.messages
            .iter()
            .map(|msg| msg.to_dict())
            .collect::<Result<Vec<Value>, serde_json::Error>>()?;

        request_body["input"] = serde_json::json!(messages_json);
        serde_json::to_string_pretty(&request_body)
    }

    /// Build as Value (for programmatic use, not just curl)
    pub fn build_request_value(&self) -> Result<Value, serde_json::Error> {
        let mut request_body = self.configuration.to_dict()?;
        
        let messages_json: Vec<Value> = self.messages
            .iter()
            .map(|msg| msg.to_dict())
            .collect::<Result<Vec<Value>, serde_json::Error>>()?;

        request_body["input"] = serde_json::json!(messages_json);
        Ok(request_body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_curl_data_openai_format() {
        let config = ModelRequestConfiguration::with_model("gpt-4.1");
        let mut input_data = ResponsesAPIInputData::new(config, Vec::new());
        
        // Add a single message
        input_data.add_message(Message::user(
            "Tell me a three sentence bedtime story about a unicorn."));
        
        let curl_data = input_data.build_curl_data().unwrap();
        
        // Should contain model
        assert!(curl_data.contains("\"model\":\"gpt-4.1\""));
        // Should contain input as array
        assert!(curl_data.contains("\"input\""));
        assert!(curl_data.contains("\"role\":\"user\""));
        assert!(curl_data.contains(
            "Tell me a three sentence bedtime story about a unicorn"));

        // Uncomment to print the curl data
        //println!("OpenAI curl -d data:\n{}", curl_data);
    }

    #[test]
    fn test_build_curl_data_xai_format() {
        let config = ModelRequestConfiguration::with_model("grok-4");
        let mut input_data = ResponsesAPIInputData::new(config, Vec::new());
        
        // Set previous_response_id
        input_data.set_previous_response_id("The previous response id");
        
        // Add message
        input_data.add_message(Message::user("What is the meaning of 42?"));
        
        let curl_data = input_data.build_curl_data().unwrap();
        
        // Should contain model
        assert!(curl_data.contains("\"model\":\"grok-4\""));
        // Should contain previous_response_id
        assert!(curl_data.contains(
            "\"previous_response_id\":\"The previous response id\""));
        // Should contain input as array
        assert!(curl_data.contains("\"input\""));
        assert!(curl_data.contains("\"role\":\"user\""));
        assert!(curl_data.contains("What is the meaning of 42?"));

        // Uncomment to print the curl data
        //println!("xAI curl -d data:\n{}", curl_data);
    }

    #[test]
    fn test_build_curl_data_multiple_messages() {
        let config = ModelRequestConfiguration::with_model("gpt-4");
        let mut input_data = ResponsesAPIInputData::new(config, Vec::new());

        input_data.add_message(Message::system("You are a helpful assistant."));
        input_data.add_message(Message::user("Hello!"));
        input_data.add_message(Message::assistant("Hi there!"));

        let curl_data = input_data.build_curl_data().unwrap();

        // Should have all three messages in input array
        assert!(curl_data.contains("\"input\""));
        // Count occurrences of "role" to verify multiple messages
        let role_count = curl_data.matches("\"role\"").count();
        assert_eq!(role_count, 3);

        // Uncomment to print the curl data
        //println!("Multiple messages curl -d data:\n{}", curl_data);
    }

    #[test]
    fn test_build_curl_data_with_configuration_fields() {
        let config = ModelRequestConfiguration::with_model("gpt-4")
            .with_temperature(0.7)
            .with_max_tokens(1000);

        let mut input_data = ResponsesAPIInputData::new(config, Vec::new());
        input_data.add_message(Message::user("Test"));

        let curl_data = input_data.build_curl_data().unwrap();
        
        // Should include configuration fields
        assert!(curl_data.contains("\"temperature\":0.7"));
        assert!(curl_data.contains("\"max_tokens\":1000"));

        // Uncomment to print the curl data
        //println!("With config fields:\n{}", curl_data);
    }

    #[test]
    fn test_clear_messages() {
        let config = ModelRequestConfiguration::with_model("gpt-4");
        let mut input_data = ResponsesAPIInputData::new(config, Vec::new());

        input_data.add_message(Message::user("Message 1"));
        input_data.add_message(Message::user("Message 2"));
        assert_eq!(input_data.messages().len(), 2);

        input_data.clear_messages();
        assert_eq!(input_data.messages().len(), 0);
    }

    #[test]
    fn test_previous_response_id() {
        let config = ModelRequestConfiguration::with_model("grok-4");
        let mut input_data = ResponsesAPIInputData::new(config, Vec::new());

        input_data.set_previous_response_id("resp_abc123");
        input_data.add_message(Message::user("Follow up"));

        let curl_data = input_data.build_curl_data().unwrap();
        assert!(curl_data.contains("\"previous_response_id\":\"resp_abc123\""));

        input_data.clear_previous_response_id();
        let curl_data2 = input_data.build_curl_data().unwrap();
        assert!(!curl_data2.contains("previous_response_id"));
    }

    #[test]
    fn test_pretty_curl_payload_example() {
        let config = ModelRequestConfiguration::with_model("grok-4")
            .with_max_tokens(500u32)
            .with_temperature(0.7f64);
        let input_data = ResponsesAPIInputData::new(
            config,
            vec![
                Message::system("You are Grok, built by xAI."),
                Message::user("Explain quantum computing in simple terms."),
            ],
        );

        let pretty_payload = input_data.build_curl_data_pretty().unwrap();
        // Uncomment to print the pretty payload
        // println!(
        //     "=== Pretty JSON for xAI-like curl -d payload ===\n{}\n=== End of payload ===",
        //     pretty_payload);

        // Verify structure
        assert!(pretty_payload.contains("\"model\": \"grok-4\""));
        assert!(pretty_payload.contains("\"max_tokens\": 500"));
        assert!(pretty_payload.contains("\"temperature\": 0.7"));
        assert!(pretty_payload.contains("\"input\":"));
        // // system and user
        assert!(pretty_payload.matches("\"role\"").count() == 2);
        assert!(pretty_payload.contains("You are Grok"));
        assert!(pretty_payload.contains("quantum computing"));
    }
}
