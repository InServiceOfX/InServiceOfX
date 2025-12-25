use crate::clients::response_api::response_api_input_data::ResponsesAPIInputData;
use crate::configurations::ModelRequestConfiguration;
use crate::messages::basic_messages::Message;
use serde_json;
use serde_json::Value;

/// Trait defining common behavior for Responses API clients
/// This allows us to define implementation once and reuse it
pub trait ResponsesAPIClientTrait {
    /// Get a mutable reference to the underlying ResponsesApiClient
    /// This allows the trait to access the base client's methods
    fn client_mut(&mut self) -> &mut ResponsesAPIClient;

    /// Get a reference to the underlying ResponsesApiClient
    fn client(&self) -> &ResponsesAPIClient;

    // ========== Pass-through methods (defined once in trait) ==========

    /// Add a message to the list
    fn add_message(&mut self, message: Message) {
        self.client_mut().add_message(message);
    }

    /// Add multiple messages
    fn add_messages(&mut self, messages: Vec<Message>) {
        self.client_mut().add_messages(messages);
    }

    /// Clear all messages
    fn clear_messages(&mut self) {
        self.client_mut().clear_messages();
    }

    /// Set the previous_response_id in configuration
    fn set_previous_response_id(&mut self, previous_response_id: impl Into<String>) {
        self.client_mut().set_previous_response_id(previous_response_id);
    }

    /// Clear the previous_response_id
    fn clear_previous_response_id(&mut self) {
        self.client_mut().clear_previous_response_id();
    }

    /// Get a reference to the configuration
    fn configuration(&self) -> &ModelRequestConfiguration {
        self.client().configuration()
    }

    /// Get a mutable reference to the configuration
    fn configuration_mut(&mut self) -> &mut ModelRequestConfiguration {
        self.client_mut().configuration_mut()
    }

    /// Get a reference to the messages
    fn messages(&self) -> &Vec<Message> {
        self.client().messages()
    }

    //--------------------------------------------------------------------------
    /// TODO: While we keep the curl command, consider if reqwest should
    /// supersede it.
    //--------------------------------------------------------------------------

    /// Build the complete curl command as a string
    fn build_curl_command(&self) -> Result<String, serde_json::Error> {
        self.client().build_curl_command()
    }

    /// Build just the JSON data for curl -d flag
    fn build_curl_data(&self) -> Result<String, serde_json::Error> {
        self.client().build_curl_data()
    }

    /// Build the JSON data in pretty format
    fn build_curl_data_pretty(&self) -> Result<String, serde_json::Error> {
        self.client().build_curl_data_pretty()
    }

    /// Send the HTTP request using reqwest (async)
    /// 
    /// This performs the same operation as the curl command would,
    /// but actually executes the request and returns the response.
    /// 
    /// # Returns
    /// The response body as a JSON Value, or an error if the request fails
    async fn send_request(&self) -> Result<
        Value,
        Box<dyn std::error::Error + Send + Sync>>
    {
        self.client().send_request().await
    }

    /// Send the HTTP request using reqwest (blocking/synchronous)
    /// 
    /// This performs the same operation as the curl command would,
    /// but actually executes the request and returns the response.
    /// Uses reqwest's blocking client for synchronous execution.
    /// 
    /// # Returns
    /// The response body as a JSON Value, or an error if the request fails
    fn send_blocking_request(&self) -> Result<
        Value,
        Box<dyn std::error::Error + Send + Sync>>
    {
        self.client().send_blocking_request()
    }
}

/// Base Responses API client
/// Uses composition to hold API key, URL, timeout, and input data
pub struct ResponsesAPIClient {
    /// API key for authentication (typically loaded from environment)
    api_key: String,
    
    /// Base URL for the API endpoint (e.g., "https://api.openai.com/v1/responses")
    base_url: String,
    
    /// Maximum time for request in seconds (for curl -m flag)
    /// Typical value includes 3600 seconds (1 hour) as suggested by xAI
    /// documentation
    timeout_seconds: Option<u32>,
    
    /// Input data (configuration + messages)
    /// This is the persistent state that can be modified between calls
    input_data: ResponsesAPIInputData,
}

impl ResponsesAPIClient {
    /// Create a new client with API key
    /// 
    /// # Arguments
    /// * `api_key` - API key for authentication
    /// * `base_url` - Base URL for the API endpoint
    /// * `timeout_seconds` - Optional timeout in seconds (default: None, use
    /// 3600 for xAI)
    /// * `configuration` - Optional model request configuration
    pub fn new(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        timeout_seconds: Option<u32>,
        configuration: Option<ModelRequestConfiguration>,
    ) -> Self {
        let config = configuration.unwrap_or_default();
        let input_data = ResponsesAPIInputData::new(config, Vec::new());

        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            timeout_seconds,
            input_data,
        }
    }

    /// Get the base URL
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Get the API key (for use in Authorization header)
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Get timeout in seconds
    pub fn timeout_seconds(&self) -> Option<u32> {
        self.timeout_seconds
    }

    /// Set timeout in seconds
    pub fn set_timeout(&mut self, timeout_seconds: Option<u32>) {
        self.timeout_seconds = timeout_seconds;
    }

    // ========== Methods that work with input_data ==========

    /// Get a reference to the input data
    pub fn input_data(&self) -> &ResponsesAPIInputData {
        &self.input_data
    }

    /// Get a mutable reference to the input data
    pub fn input_data_mut(&mut self) -> &mut ResponsesAPIInputData {
        &mut self.input_data
    }

    /// Get a reference to the configuration
    pub fn configuration(&self) -> &ModelRequestConfiguration {
        self.input_data.configuration()
    }

    /// Get a mutable reference to the configuration
    pub fn configuration_mut(&mut self) -> &mut ModelRequestConfiguration {
        self.input_data.configuration_mut()
    }

    /// Get a reference to the messages
    pub fn messages(&self) -> &Vec<Message> {
        self.input_data.messages()
    }

    /// Add a message to the list
    pub fn add_message(&mut self, message: Message) {
        self.input_data.add_message(message);
    }

    /// Add multiple messages
    pub fn add_messages(&mut self, messages: Vec<Message>) {
        self.input_data.add_messages(messages);
    }

    /// Clear all messages
    pub fn clear_messages(&mut self) {
        self.input_data.clear_messages();
    }

    /// Set the previous_response_id in configuration
    pub fn set_previous_response_id(
        &mut self,
        previous_response_id: impl Into<String>) 
    {
        self.input_data.set_previous_response_id(previous_response_id);
    }

    /// Clear the previous_response_id
    pub fn clear_previous_response_id(&mut self) {
        self.input_data.clear_previous_response_id();
    }

    // ========== Curl command building ==========

    /// Build the complete curl command as a string
    /// 
    /// This constructs the full curl command including:
    /// - URL
    /// - Content-Type header (hardcoded as "application/json")
    /// - Authorization header with API key
    /// - Timeout flag (-m) if specified
    /// - Data flag (-d) with the JSON request body
    pub fn build_curl_command(&self) -> Result<String, serde_json::Error> {
        let curl_data = self.input_data.build_curl_data()?;

        let mut curl_cmd = format!("curl {} \\\n", self.base_url);
        curl_cmd.push_str("  -H \"Content-Type: application/json\" \\\n");
        curl_cmd.push_str(
            &format!("  -H \"Authorization: Bearer {}\" \\\n", self.api_key));

        // Add timeout if specified (in seconds, for curl -m flag)
        if let Some(timeout) = self.timeout_seconds {
            curl_cmd.push_str(&format!("  -m {} \\\n", timeout));
        }

        curl_cmd.push_str(&format!("  -d '{}'", curl_data));

        Ok(curl_cmd)
    }

    /// Build just the JSON data for curl -d flag
    pub fn build_curl_data(&self) -> Result<String, serde_json::Error> {
        self.input_data.build_curl_data()
    }

    /// Build the JSON data in pretty format
    pub fn build_curl_data_pretty(&self) -> Result<String, serde_json::Error> {
        self.input_data.build_curl_data_pretty()
    }

    /// Send the HTTP request using reqwest (async)
    /// 
    /// This performs the same operation as the curl command would,
    /// but actually executes the request and returns the response.
    /// 
    /// # Returns
    /// The response body as a JSON Value, or an error if the request fails
    pub async fn send_request(&self) -> Result<
        Value,
        Box<dyn std::error::Error + Send + Sync>>
    {
        // Build the request body
        let request_body = self.input_data.build_request_value()?;
        
        // Create reqwest client with timeout if specified
        let mut client_builder = reqwest::Client::builder();
        if let Some(timeout_secs) = self.timeout_seconds {
            client_builder = client_builder.timeout(
                std::time::Duration::from_secs(timeout_secs as u64));
        }
        let client = client_builder.build()?;
        
        // Build the request
        let request = client
            .post(&self.base_url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request_body);
        
        // Send the request
        let response = request.send().await?;
        
        // Check for HTTP errors
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(
                |_| "Unknown error".to_string());
            return Err(format!("HTTP {}: {}", status, error_text).into());
        }

        // Parse JSON response
        let json_response: Value = response.json().await?;
        Ok(json_response)
    }

    /// Send the HTTP request using reqwest (blocking/synchronous)
    /// 
    /// This performs the same operation as the curl command would,
    /// but actually executes the request and returns the response.
    /// Uses reqwest's blocking client for synchronous execution.
    /// 
    /// # Returns
    /// The response body as a JSON Value, or an error if the request fails
    pub fn send_blocking_request(&self) -> Result<
        Value,
        Box<dyn std::error::Error + Send + Sync>>
    {
        // Build the request body
        let request_body = self.input_data.build_request_value()?;
        
        // Create reqwest blocking client with timeout if specified
        let mut client_builder = reqwest::blocking::Client::builder();
        if let Some(timeout_secs) = self.timeout_seconds {
            client_builder = client_builder.timeout(
                std::time::Duration::from_secs(timeout_secs as u64));
        }
        let client = client_builder.build()?;
        
        // Build the request
        let request = client
            .post(&self.base_url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request_body);
        
        // Send the request
        let response = request.send()?;
        
        // Check for HTTP errors
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().unwrap_or_else(
                |_| "Unknown error".to_string());
            return Err(format!("HTTP {}: {}", status, error_text).into());
        }
        
        // Parse JSON response
        let json_response: Value = response.json()?;
        Ok(json_response)
    }
}

/// OpenAI Responses API client
/// Specialized client with hardcoded OpenAI URL
pub struct OpenAIResponsesClient {
    client: ResponsesAPIClient,
}

impl ResponsesAPIClientTrait for OpenAIResponsesClient {
    fn client_mut(&mut self) -> &mut ResponsesAPIClient {
        &mut self.client
    }

    fn client(&self) -> &ResponsesAPIClient {
        &self.client
    }
}

impl OpenAIResponsesClient {
    /// Create a new OpenAI Responses API client
    /// Loads OPENAI_API_KEY from environment if api_key not provided
    /// 
    /// # Arguments
    /// * `api_key` - Optional API key value; if None, loads from OPENAI_API_KEY
    ///  env var
    /// * `timeout_seconds` - Optional timeout (OpenAI doesn't require it, but
    /// can be set)
    /// * `configuration` - Optional model request configuration
    pub fn new(
        api_key: Option<impl Into<String>>,
        timeout_seconds: Option<u32>,
        configuration: Option<ModelRequestConfiguration>,
    ) -> Result<Self, String> {
        use core_code::utilities::load_environment_file::get_environment_variable;
        
        let api_key = if let Some(key) = api_key {
            key.into()
        } else {
            get_environment_variable("OPENAI_API_KEY")
                .map_err(|e| format!(
                    "Failed to get OPENAI_API_KEY from environment: {}", e))?
        };
        
        let client = ResponsesAPIClient::new(
                api_key,
                "https://api.openai.com/v1/responses",
                timeout_seconds,
                configuration,
        );
        
        Ok(Self {
            client,
        })
    }
}

/// xAI/Grok Responses API client
/// Specialized client with hardcoded xAI URL and default timeout
pub struct GrokResponsesClient {
    client: ResponsesAPIClient,
}

impl ResponsesAPIClientTrait for GrokResponsesClient {
    fn client_mut(&mut self) -> &mut ResponsesAPIClient {
        &mut self.client
    }

    fn client(&self) -> &ResponsesAPIClient {
        &self.client
    }
}

impl GrokResponsesClient {
    /// Create a new xAI/Grok Responses API client
    /// Loads XAI_API_KEY from environment if api_key not provided
    /// 
    /// # Arguments
    /// * `api_key` - Optional API key value; if None, loads from XAI_API_KEY
    /// env var
    /// * `timeout_seconds` - Optional timeout in seconds (default: 3600 as
    /// suggested by xAI docs)
    /// * `configuration` - Optional model request configuration
    pub fn new(
        api_key: Option<impl Into<String>>,
        timeout_seconds: Option<u32>,
        configuration: Option<ModelRequestConfiguration>,
    ) -> Result<Self, String> {
        use core_code::utilities::load_environment_file::get_environment_variable;
        
        let api_key = if let Some(key) = api_key {
            key.into()
        } else {
            get_environment_variable("XAI_API_KEY")
                .map_err(|e| format!(
                    "Failed to get XAI_API_KEY from environment: {}", e))?
        };

        // Default to 3600 seconds (1 hour) as suggested by xAI documentation
        let timeout = timeout_seconds.unwrap_or(3600);
        
        let client = ResponsesAPIClient::new(
                api_key,
                "https://api.x.ai/v1/responses",
                Some(timeout),
                configuration,
        );
        
        Ok(Self {
            client,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_client_build_curl_command() {
        let config = ModelRequestConfiguration::with_model("gpt-4.1");
        let mut client = OpenAIResponsesClient::new(
            Some("dummy-openai-key".to_string()),
            None,
            Some(config),
        ).unwrap();

        client.add_message(Message::user(
            "Tell me a three sentence bedtime story about a unicorn."));

        let curl_cmd = client.build_curl_command().unwrap();

        // Uncomment to print the curl command
        // println!("\n{}", "=".repeat(80));
        // println!("OpenAI Responses API - Full curl command");
        // println!("{}", "=".repeat(80));
        // println!("{}", curl_cmd);
        // println!("{}", "=".repeat(80));
        
        assert!(curl_cmd.contains("api.openai.com/v1/responses"));
        assert!(curl_cmd.contains("Content-Type: application/json"));
        assert!(curl_cmd.contains("Authorization: Bearer"));
        assert!(curl_cmd.contains("\"model\":\"gpt-4.1\""));
        assert!(curl_cmd.contains("-d"));
    }

    #[test]
    fn test_xai_client_build_curl_command() {
        let config = ModelRequestConfiguration::with_model("grok-4");
        let mut client = GrokResponsesClient::new(
            Some("dummy-xai-key".to_string()),
            None,
            Some(config)).unwrap();

        client.set_previous_response_id("The previous response id");
        client.add_message(Message::system(
            "You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy."));
        client.add_message(Message::user(
            "What is the meaning of life, the universe, and everything?"));
        
        let curl_cmd = client.build_curl_command().unwrap();

        // Uncomment to print the curl command
        // println!("\n{}", "=".repeat(80));
        // println!("xAI/Grok Responses API - Full curl command");
        // println!("{}", "=".repeat(80));
        // println!("{}", curl_cmd);
        // println!("{}", "=".repeat(80));
        
        assert!(curl_cmd.contains("api.x.ai/v1/responses"));
        assert!(curl_cmd.contains("Content-Type: application/json"));
        assert!(curl_cmd.contains("Authorization: Bearer"));
        assert!(curl_cmd.contains("-m 3600")); // Default timeout
        assert!(curl_cmd.contains("\"model\":\"grok-4\""));
        assert!(curl_cmd.contains("\"previous_response_id\""));
        assert!(curl_cmd.contains("-d"));
    }

    #[test]
    fn test_trait_methods_work_for_both_clients() {
        // Test that trait methods work for OpenAI
        let mut openai_client = OpenAIResponsesClient::new(
            Some("dummy-openai-key".to_string()), None, None).unwrap();
        openai_client.add_message(Message::user("Test"));
        assert_eq!(openai_client.messages().len(), 1);
        openai_client.clear_messages();
        assert_eq!(openai_client.messages().len(), 0);

        // Test that trait methods work for xAI
        let mut xai_client = GrokResponsesClient::new(
            Some("dummy-xai-key".to_string()), None, None).unwrap();
        xai_client.add_message(Message::user("Test"));
        assert_eq!(xai_client.messages().len(), 1);
        xai_client.clear_messages();
        assert_eq!(xai_client.messages().len(), 0);
    }
}