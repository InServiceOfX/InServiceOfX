use crate::clients::response_api::http_clients::{
    BlockingReqwestClient,
    ReqwestClient};
use crate::clients::response_api::response_api_client::ResponsesAPIClient;
use crate::clients::response_api::response_object::ResponseObject;
use crate::configurations::ModelRequestConfiguration;
use crate::messages::basic_messages::Message;
use serde_json::Value;
use std::error::Error;

pub trait ResponsesAndReqwestTrait
{
    fn client_mut(&mut self) -> &mut ResponsesAndReqwestClient;
    fn client(&self) -> &ResponsesAndReqwestClient;

    fn send_request(&self) -> impl std::future::Future<Output = Result<
        Value,
        Box<dyn std::error::Error + Send + Sync>>> + Send where Self: Sync {
        async {
            self.client().send_request().await
        }
    }

    fn send_request_and_parse(&self)
        -> impl Future<
            Output = Result<
                ResponseObject,
                Box<dyn Error + Send + Sync>>> + Send where Self: Sync
    {
        async {
            self.client().send_request_and_parse().await
        }
    }

    fn configuration(&self) -> &ModelRequestConfiguration {
        self.client().configuration()
    }

    fn configuration_mut(&mut self) -> &mut ModelRequestConfiguration {
        self.client_mut().configuration_mut()
    }

    fn add_message(&mut self, message: Message) {
        self.client_mut().add_message(message);
    }

    fn add_messages(&mut self, messages: Vec<Message>) {
        self.client_mut().add_messages(messages);
    }

    fn clear_messages(&mut self) {
        self.client_mut().clear_messages();
    }

    fn set_previous_response_id(
        &mut self,
        previous_response_id: impl Into<String>)
    {
        self.client_mut().set_previous_response_id(previous_response_id);
    }
}

pub struct ResponsesAndReqwestClient {
    pub responses_client: ResponsesAPIClient,
    reqwest_client: ReqwestClient,
}

impl ResponsesAndReqwestClient {
    pub fn new(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        timeout_seconds: Option<u32>,
        configuration: Option<ModelRequestConfiguration>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>>
    {
        let responses_client = ResponsesAPIClient::new(
            api_key, base_url, timeout_seconds, configuration)?;
        let reqwest_client = ReqwestClient::new(timeout_seconds)?;

        Ok(Self { responses_client, reqwest_client })
    }

    pub fn configuration(&self) -> &ModelRequestConfiguration {
        self.responses_client.configuration()
    }

    pub fn configuration_mut(&mut self) -> &mut ModelRequestConfiguration {
        self.responses_client.configuration_mut()
    }

    pub fn add_message(&mut self, message: Message) {
        self.responses_client.add_message(message);
    }

    pub fn add_messages(&mut self, messages: Vec<Message>) {
        self.responses_client.add_messages(messages);
    }
    
    pub fn clear_messages(&mut self) {
        self.responses_client.clear_messages();
    }

    pub fn set_previous_response_id(
        &mut self,
        previous_response_id: impl Into<String>)
    {
        self.responses_client.set_previous_response_id(previous_response_id);
    }

    pub async fn send_request(&self) -> Result<
        Value,
        Box<dyn std::error::Error + Send + Sync>>
    {
        let request_body = 
            self.responses_client.input_data().build_request_value()?;
        let request = self.reqwest_client.client
            .post(&self.responses_client.base_url().to_string())
            .header("Content-Type", "application/json")
            .header(
                "Authorization",
                format!("Bearer {}", self.responses_client.api_key()))
            .json(&request_body);

        let response = request.send().await?;

        // Check for HTTP errors
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(
                |_| "Unknown error".to_string());

            // Check specifically for rate limiting
            if status.as_u16() == 429 {
                eprintln!(
                    "ERROR: API RATE LIMITED (429)! Response: {}",
                    error_text
                );
                return Err(format!(
                    "RATE LIMITED: API returned 429. Response: {}",
                    error_text
                ).into());
            }

            return Err(format!(
                "API request failed with status {}: {}",
                status, error_text
            ).into());        }

        // Parse JSON response
        let json_response: Value = response.json().await?;
        Ok(json_response)
    }

    pub fn send_request_and_parse(&self)
        -> impl Future<
            Output = Result<
                ResponseObject,
                Box<dyn Error + Send + Sync>>> + Send where Self: Sync
    {
        async {
            // Convert raw Value to ResponseObject
            let raw_value = self.send_request().await?;
            ResponseObject::from_value(raw_value).map_err(|e| Box::new(e) as _)
        }
    }
}

pub trait ResponsesAndBlockingReqwestTrait
{
    fn client_mut(&mut self) -> &mut ResponsesAndBlockingReqwestClient;
    fn client(&self) -> &ResponsesAndBlockingReqwestClient;

    /// Send the HTTP request using reqwest (blocking/synchronous)
    /// 
    /// This performs the same operation as the curl command would,
    /// but actually executes the request and returns the response.
    /// Uses reqwest's blocking client for synchronous execution.
    /// 
    /// # Returns
    /// The response body as a JSON Value, or an error if the request fails
    fn send_request(&self) -> Result<
        Value,
        Box<dyn std::error::Error + Send + Sync>>
    {
        self.client().send_request()
    }

    /// Send request and parse to structured ResponseObject
    /// (shortcut for raw + from_value)
    fn send_request_and_parse(&self)
        -> Result<
            ResponseObject,
            Box<dyn std::error::Error + Send + Sync>>
    {
        self.client().send_request_and_parse()
    }

    fn configuration(&self) -> &ModelRequestConfiguration {
        self.client().configuration()
    }

    fn configuration_mut(&mut self) -> &mut ModelRequestConfiguration {
        self.client_mut().configuration_mut()
    }

    fn add_message(&mut self, message: Message) {
        self.client_mut().add_message(message);
    }

    fn add_messages(&mut self, messages: Vec<Message>) {
        self.client_mut().add_messages(messages);
    }

    fn clear_messages(&mut self) {
        self.client_mut().clear_messages();
    }

    fn set_previous_response_id(
        &mut self,
        previous_response_id: impl Into<String>)
    {
        self.client_mut().set_previous_response_id(previous_response_id);
    }
}

pub struct ResponsesAndBlockingReqwestClient {
    pub responses_client: ResponsesAPIClient,
    reqwest_client: BlockingReqwestClient,
}

impl ResponsesAndBlockingReqwestClient {
    pub fn new(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        timeout_seconds: Option<u32>,
        configuration: Option<ModelRequestConfiguration>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>>
    {
        let responses_client = ResponsesAPIClient::new(
            api_key, base_url, timeout_seconds, configuration)?;
        let reqwest_client = BlockingReqwestClient::new(timeout_seconds)?;

        Ok(Self { responses_client, reqwest_client })
    }

    pub fn configuration(&self) -> &ModelRequestConfiguration {
        self.responses_client.configuration()
    }

    pub fn configuration_mut(&mut self) -> &mut ModelRequestConfiguration {
        self.responses_client.configuration_mut()
    }

    pub fn add_message(&mut self, message: Message) {
        self.responses_client.add_message(message);
    }

    pub fn add_messages(&mut self, messages: Vec<Message>) {
        self.responses_client.add_messages(messages);
    }
    
    pub fn clear_messages(&mut self) {
        self.responses_client.clear_messages();
    }

    pub fn set_previous_response_id(
        &mut self,
        previous_response_id: impl Into<String>)
    {
        self.responses_client.set_previous_response_id(previous_response_id);
    }

    /// Send the HTTP request using reqwest (blocking/synchronous)
    /// 
    /// This performs the same operation as the curl command would,
    /// but actually executes the request and returns the response.
    /// Uses reqwest's blocking client for synchronous execution.
    /// 
    /// # Returns
    /// The response body as a JSON Value, or an error if the request fails    
    pub fn send_request(&self) -> Result<
        Value,
        Box<dyn std::error::Error + Send + Sync>>
    {
        // Build the request body
        let request_body = 
            self.responses_client.input_data().build_request_value()?;

        let request = self.reqwest_client.client
            .post(&self.responses_client.base_url().to_string())
            .header("Content-Type", "application/json")
            .header(
                "Authorization",
                format!("Bearer {}", self.responses_client.api_key()))
            .json(&request_body);

        let response = request.send()?;

        // Check for HTTP errors
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().unwrap_or_else(
                |_| "Unknown error".to_string());

            // Check specifically for rate limiting
            if status.as_u16() == 429 {
                eprintln!(
                    "ERROR: API RATE LIMITED (429)! Response: {}",
                    error_text
                );
                return Err(format!(
                    "RATE LIMITED: API returned 429. Response: {}",
                    error_text
                ).into());
            }

            return Err(format!(
                "API request failed with status {}: {}",
                status, error_text
            ).into());        }

        // Parse JSON response
        let json_response: Value = response.json()?;
        Ok(json_response)
    }

    pub fn send_request_and_parse(&self)
        -> Result<
            ResponseObject,
            Box<dyn Error + Send + Sync>>
    {
        // Convert raw Value to ResponseObject
        let raw_value = self.send_request()?;
        ResponseObject::from_value(raw_value).map_err(|e| Box::new(e) as _)
    }
}