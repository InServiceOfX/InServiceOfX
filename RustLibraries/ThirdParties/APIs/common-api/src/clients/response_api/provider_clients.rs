// Bring the trait into scope with pub use.
pub use crate::clients::response_api::base_composed_clients::ResponsesAndReqwestTrait;

use crate::clients::response_api::base_composed_clients::ResponsesAndReqwestClient;
use crate::configurations::ModelRequestConfiguration;

/// OpenAI Responses API client
/// Specialized client with hardcoded OpenAI URL
pub struct OpenAIResponsesClient {
    client: ResponsesAndReqwestClient,
}

impl ResponsesAndReqwestTrait for OpenAIResponsesClient {
    fn client_mut(&mut self) -> &mut ResponsesAndReqwestClient {
        &mut self.client
    }

    fn client(&self) -> &ResponsesAndReqwestClient {
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
        
        let client = ResponsesAndReqwestClient::new(
            api_key,
            "https://api.openai.com/v1/responses",
            timeout_seconds,
            configuration,
        ).unwrap();
        
        Ok(Self { client })
    }
}

/// xAI/Grok Responses API client
/// Specialized client with hardcoded xAI URL and default timeout
pub struct GrokResponsesClient {
    client: ResponsesAndReqwestClient,
}

impl ResponsesAndReqwestTrait for GrokResponsesClient {
    fn client_mut(&mut self) -> &mut ResponsesAndReqwestClient {
        &mut self.client
    }

    fn client(&self) -> &ResponsesAndReqwestClient {
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
        let client = ResponsesAndReqwestClient::new(
            api_key,
            "https://api.x.ai/v1/responses",
            timeout_seconds,
            configuration,
        ).unwrap();
        
        Ok(Self { client })
    }
}

pub struct GroqResponsesClient {
    client: ResponsesAndReqwestClient,
}

impl ResponsesAndReqwestTrait for GroqResponsesClient {
    fn client_mut(&mut self) -> &mut ResponsesAndReqwestClient {
        &mut self.client
    }

    fn client(&self) -> &ResponsesAndReqwestClient {
        &self.client
    }
}

impl GroqResponsesClient {
    //--------------------------------------------------------------------------
    /// https://console.groq.com/docs/responses-api
    //--------------------------------------------------------------------------
    pub fn new(
        api_key: Option<impl Into<String>>,
        timeout_seconds: Option<u32>,
        configuration: Option<ModelRequestConfiguration>,
    ) -> Result<Self, String> {
        use core_code::utilities::load_environment_file::get_environment_variable;
        
        let api_key = if let Some(key) = api_key {
            key.into()
        } else {
            get_environment_variable("GROQ_API_KEY")
                .map_err(|e| format!(
                    "Failed to get GROQ_API_KEY from environment: {}", e))?
        };

        // Default to 3600 seconds (1 hour) as suggested by xAI documentation
        let client = ResponsesAndReqwestClient::new(
            api_key,
            "https://api.groq.com/openai/v1",
            timeout_seconds,
            configuration,
        ).unwrap();
        
        Ok(Self { client })
    }
}