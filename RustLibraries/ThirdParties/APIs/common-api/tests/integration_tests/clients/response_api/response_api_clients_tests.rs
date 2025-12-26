// Simple test as requested.
#[test]
//#[ignore = "Integration test; run with cargo test -- --ignored to execute"]
fn simple_integration_test() {
    assert_eq!(1 + 1, 2);
    // In full integration tests, add e.g.:
    // use common_api::clients::response_api::response_api_client::*;
    // let client = ...;
    // assert!(client.build_curl_command().is_ok());
    // Test real API response parsing, e.g., let resp: ResponseObject = serde_json::from_str(&json_str)?; assert!(!resp.id.is_empty());
}

#[cfg(test)]
mod tests {

    use common_api::clients::response_api::response_api_clients::*;
    use common_api::configurations::model_request_configuration::*;
    use core_code::utilities::load_environment_file::*;

    #[test]
    fn test_grok_client_send_request()
    {
        load_environment_file_from_default_path();
        let configuration = ModelRequestConfiguration::with_model(
            "grok-4-1-fast-reasoning");

        let mut client = GrokResponsesClient::new(
            Some(get_environment_variable_unwrap("XAI_API_KEY")),
            Some(3600),
            Some(configuration)).unwrap();

        let response = client.send_request().await.unwrap();
        // let response_object = ResponseObject::from_json(&response.to_string()).unwrap();
        // assert!(!response_object.id.is_empty());
    }
}