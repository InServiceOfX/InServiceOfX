//------------------------------------------------------------------------------
/// Example USAGE:
/// cargo test tools::api_tools_tests --test integration_tests -- --no-capture
/// Run an individual test:
/// cargo test --test integration_tests test_grok_client_works_with_code_interpreter -- --no-capture
//------------------------------------------------------------------------------

#[cfg(test)]
mod tests {

    use common_api::clients::response_api::provider_clients::*;
    use common_api::configurations::model_request_configuration::*;
    use common_api::messages::basic_messages::Message;
    use common_api::tools::api_tools::*;
    use common_api::tools::tool_choice::*;
    use core_code::utilities::load_environment_file::*;

    //--------------------------------------------------------------------------
    /// https://docs.x.ai/docs/guides/tools/code-execution-tool
    //--------------------------------------------------------------------------

    #[test]
    fn test_grok_client_works_with_code_interpreter()
    {
        load_environment_file_from_default_path();
        let mut configuration = ModelRequestConfiguration::with_model(
            "grok-4-1-fast-reasoning");

        configuration.tool_choice = Some(ToolChoiceMode::Required.to_value());
        configuration.tools = Some(vec![APITools::CodeInterpreter {
            container: CodeInterpreterContainer::default(),
        }.to_value()]);
        let mut client = GrokResponsesBlockingClient::new(
            Some(get_environment_variable_unwrap("XAI_API_KEY")),
            None,
            Some(configuration)).unwrap();

        let user_prompt = format!(
            "Calculate the compound interest for $10,000 at 5% annually for {}",
            "10 years");
        client.add_message(Message::user(user_prompt));

        let response = client.send_request_and_parse().unwrap();

        assert!(response.output.len() > 0);
        assert_eq!(response.output.len(), 2);
        assert_eq!(response.output[0].type_, "code_interpreter_call");
        assert_eq!(response.output[0].role, "");
        assert_eq!(response.output[1].type_, "message");
        assert_eq!(response.output[1].role, "assistant");

        assert_eq!(response.output[1].content[0].type_, "output_text");
        assert!(response.output[1].content[0].text.contains("6,288"));

        // Example (actual) response:
        // **The compound interest earned is $6,288.95.**

        // ### Breakdown:
        // - **Principal (P)**: $10,000
        // - **Annual interest rate (r)**: 5% or 0.05
        // - **Compounding frequency (n)**: Annually (1 time per year)
        // - **Time (t)**: 10 years
        
        // The formula for the future value (A) with compound interest is:  
        // **A = P × (1 + r/n)^(n×t)**  
        
        // Plugging in the values:  
        // **A = 10,000 × (1 + 0.05/1)^(1×10) = 10,000 × (1.05)^10 ≈ $16,288.95**  
        
        // **Compound interest = A - P ≈ $16,288.95 - $10,000 = $6,288.95** (rounded to two decimal places).
        // Uncomment to print the response for inspection
        //println!("Response: {}", response.output[1].content[0].text);


    }

    #[tokio::test]
    async fn test_groq_client_works_with_code_interpreter()
    {
        load_environment_file_from_default_path();
        let mut configuration = ModelRequestConfiguration::with_model(
            "openai/gpt-oss-20b");

        configuration.tool_choice = Some(ToolChoiceMode::Required.to_value());
        configuration.tools = Some(vec![APITools::CodeInterpreter {
            container: CodeInterpreterContainer::default(),
        }.to_value()]);
        let mut client = GroqResponsesClient::new(
            Some(get_environment_variable_unwrap("GROQ_API_KEY")),
            None,
            Some(configuration)).unwrap();

        client.add_message(Message::user(
            "What is 1312 X 3333? Output only the final answer."
        ));

        let response = client.send_request_and_parse().await.unwrap();
        assert!(response.output.len() > 0);
        assert_eq!(response.output.len(), 3);
        assert_eq!(response.output[0].type_, "reasoning");
        assert_eq!(response.output[0].role, "");
        assert_eq!(response.output[1].type_, "mcp_call");
        assert_eq!(response.output[1].role, "");

        assert_eq!(response.output[2].type_, "message");
        assert_eq!(response.output[2].role, "assistant");
        assert_eq!(response.output[2].content[0].type_, "output_text");
        assert!(response.output[2].content[0].text.contains("4372896"));

        // Example (actual) response:
        // 4372896
        // Uncomment to print the response for inspection
        //println!("Response: {}", response.output[2].content[0].text);
    }
}