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

    //--------------------------------------------------------------------------
    /// https://docs.x.ai/docs/guides/tools/code-execution-tool#data-analysis
    //--------------------------------------------------------------------------
    #[test]
    fn test_grok_client_code_interpreter_for_data_analysis()
    {
        load_environment_file_from_default_path();
        let mut configuration = ModelRequestConfiguration::with_model(
            "grok-4-1-fast-reasoning");

        configuration.tool_choice = Some(ToolChoiceMode::Required.to_value());
        configuration.tools = Some(vec![APITools::CodeInterpreter {
            container: CodeInterpreterContainer::default(),
        }.to_value()]);

        // Obtained this error:
        // "API request failed with status 400 Bad Request: {\"code\":\"400\",\"error\":\"Argument not supported: include\"}"
        // but this API documentation says it's supported:
        // https://platform.openai.com/docs/api-reference/responses/create#responses_create-include
        // and for xAI:
        // https://docs.x.ai/docs/guides/tools/code-execution-tool#data-analysis
        // configuration.include = Some(vec![serde_json::Value::String(
        //     "verbose_streaming".to_string())]);

        let mut client = GrokResponsesBlockingClient::new(
            Some(get_environment_variable_unwrap("XAI_API_KEY")),
            None,
            Some(configuration)).unwrap();

        let user_prompt =
            r#"I have sales data for Q1-Q4: [120000, 135000, 98000, 156000].
            Please analyze this data and create a visualization showing:
            1. Quarterly trends
            2. Growth rates
            3. Statistical summary"#;
        client.add_message(Message::user(user_prompt));

        let response = client.send_request_and_parse().unwrap();
        assert!(response.output.len() > 0);
        assert_eq!(response.output.len(), 2);
        assert_eq!(response.output[0].type_, "code_interpreter_call");
        assert_eq!(response.output[0].role, "");
        assert_eq!(response.output[1].type_, "message");
        assert_eq!(response.output[1].role, "assistant");
        assert_eq!(response.output[1].content[0].type_, "output_text");

        // Example (actual) response:
        // ### Sales Data Analysis (Q1-Q4)

// #### 1. Quarterly Trends and Growth Rates
// The sales show an initial increase from Q1 to Q2 (+12.50%), a significant dip in Q3 (-27.41%), followed by a strong recovery in Q4 (+59.18%). Overall, sales trended upward despite volatility, ending 30% higher than Q1.

// **Data Table:**

// | Quarter | Sales    | Growth Rate (%) |
// |---------|----------|-----------------|
// | Q1      | 120,000 | N/A             |
// | Q2      | 135,000 | +12.50          |
// | Q3      | 98,000  | -27.41          |
// | Q4      | 156,000 | +59.18          |

// **Visual: Quarterly Sales Trend (ASCII Bar Chart)**  
// *(Scaled to max width of 50 characters for Q4)*
// ```
// Q1: ###################################### (120,000)
// Q2: ########################################### (135,000)
// Q3: ############################### (98,000)
// Q4: ################################################## (156,000)
// ```

// #### 2. Growth Rates Summary
// - Q2 vs Q1: **+12.50%**
// - Q3 vs Q2: **-27.41%**
// - Q4 vs Q3: **+59.18%**
// - Average Growth Rate: **+14.76%**

// #### 3. Statistical Summary
// ```
// Count:              4
// Mean:           127,250
// Std Dev:        24,459.15
// Min:             98,000
// 25th Percentile: 114,500
// Median:         127,500
// 75th Percentile:140,250
// Max:            156,000
// ```
// - **Total Sales**: 509,000
// - **Average Quarterly Sales**: 127,250

// **Key Insights**: Sales are volatile (high std dev relative to mean), with Q3 as an outlier low. Strong Q4 performance suggests potential seasonality or recovery from external factors. Recommend investigating Q3 dip for future forecasting.
        // Uncomment to print the response for inspection
        //println!("Response: {}", response.output[1].content[0].text);

        assert!(response.output[1].content[0].text.contains("127,250"));
        client.add_message(
            Message::assistant(response.output[1].content[0].text.clone()));

        assert!(response.tools.len() > 0);
        assert_eq!(response.tools.len(), 1);
        // Uncomment to print the response for inspection
        //println!("Response: {}", response.tools[0]);
        // Example (actual) output:
        // Usage {
        //     input_tokens: 4310,
        //     input_tokens_details: InputTokensDetails {
        //         cached_tokens: 2465,
        //     },
        //     output_tokens: 2129,
        //     output_tokens_details: OutputTokensDetails {
        //         reasoning_tokens: 1030,
        //     },
        //     total_tokens: 6439,
        // }
        //println!("Response: {:#?}", response.usage);

        assert_eq!(
            response.tools[0].get("type").and_then(|t: &_| Some(t))
                .unwrap(),
            "code_interpreter");

        // Example (actual) usage details:
        // Usage details:
//   input_tokens: 4310
//   output_tokens: 2129
//   total_tokens: 6439
//   reasoning_tokens: 1030
        // Uncomment to print the response for inspection
        // println!("\nUsage details:");
        // println!("  input_tokens: {}", response.usage.input_tokens);
        // println!("  output_tokens: {}", response.usage.output_tokens);
        // println!("  total_tokens: {}", response.usage.total_tokens);
        // println!(
        //     "  reasoning_tokens: {}",
        //     response.usage.output_tokens_details.reasoning_tokens);

        let user_prompt = "Now predict Q1 next year using linear regression";
        client.add_message(Message::user(user_prompt));

        let response = client.send_request_and_parse().unwrap();
        assert!(response.output.len() > 0);
        assert_eq!(response.output.len(), 2);
        assert_eq!(response.output[0].type_, "code_interpreter_call");
        assert_eq!(response.output[0].role, "");
        assert_eq!(response.output[1].type_, "message");
        assert_eq!(response.output[1].role, "assistant");
        assert_eq!(response.output[1].content[0].type_, "output_text");

        assert!(response.output[1].content[0].text.contains("linear regression"));
        // Example (actual) response:
        // ### Linear Regression Prediction for Q1 2026

// #### Model Summary
// Fitted linear regression model on quarters (x = [1,2,3,4]) vs. sales (y = [120,000; 135,000; 98,000; 156,000]):

// - **Equation**: `Sales = 7,100 × Quarter + 109,500`
// - **Slope**: 7,100 (average quarterly increase)
// - **Intercept**: 109,500
// - **R²**: 0.14 (low fit; model explains only ~14% of variance due to Q3 outlier/dip. Prediction is rough—consider alternatives like exponential smoothing or external factors for better accuracy)

// #### Predictions
// | Quarter | Actual Sales | Predicted Sales |
// |---------|--------------|-----------------|
// | Q1      | 120,000      | 116,600         |
// | Q2      | 135,000      | 123,700         |
// | Q3      | 98,000       | 130,800         |
// | Q4      | 156,000      | 137,900         |
// | **Q1 2026** | -         | **145,000**     |

// **Q1 2026 Prediction: 145,000**  
// (This assumes continuation of the mild upward trend, ignoring the Q3 anomaly. With Q4's strong performance, actual could be higher; low R² suggests high uncertainty ± ~25,000 based on std dev.)

// #### Updated Visualization: Actual vs. Trend Line (ASCII)
// Sales scaled to max (156k); trend line overlaid (dashes approximate fit):

// ```
// Q1: |██████████████████████████████████████░░░░░░░░░░░░| 120k (Pred: 117k)
//     └─────────────────█───────────────────────────────
// Q2: |███████████████████████████████████████████░░░░░░░| 135k (Pred: 124k)
//     └──────────────────█──────────────────────────────
// Q3: |███████████████████████████████░░░░░░░░░░░░░░░░░░░| 98k  (Pred: 131k)*
//     └───────────────────█─────────────────────────────
// Q4: |██████████████████████████████████████████████████| 156k (Pred: 138k)
//     └────────────────────█────────────────────────────
// Q1':**███████████████████████████████████████████████░░| **145k**
//     └─────────────────────█───────────────────────────
// ```
// (*Q3 underperformed trend significantly.)

// **Recommendation**: Linear regression is simplistic here due to volatility. For robustness, use time-series models (e.g., ARIMA) or incorporate seasonality if more data available. Monitor Q4 momentum for upward revision.
        // Uncomment to print the response for inspection
        //println!("Response: {}", response.output[1].content[0].text);
    }

    //--------------------------------------------------------------------------
    /// https://docs.x.ai/docs/guides/tools/code-execution-tool#common-use-cases
    //--------------------------------------------------------------------------
    #[test]
    fn test_grok_client_works_with_common_cases_for_code_interpreter()
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

        // https://docs.x.ai/docs/guides/tools/code-execution-tool#financial-analysis
        // # Portfolio optimization, risk calculations, option pricing
        let user_prompt = format!(
            "Calculate the Sharpe ratio for a portfolio with returns {}",
            "[0.12, 0.08, -0.03, 0.15] and risk-free rate 0.02"
        );
        client.add_message(Message::user(user_prompt));

        let response = client.send_request_and_parse().unwrap();
        assert!(response.output.len() > 0);
        if response.output.len() == 2 {
            assert_eq!(response.output[0].type_, "code_interpreter_call");
            assert_eq!(response.output[0].role, "");
            assert_eq!(response.output[1].type_, "message");
            assert_eq!(response.output[1].role, "assistant");
        }
        let assistant_messages = response.extract_assistant_messages();
        assert!(assistant_messages.len() > 0);
        assert_eq!(assistant_messages.len(), 1);

        // Example (actual) response:
        //  **Sharpe ratio = (Average portfolio return - Risk-free rate) / Standard deviation of portfolio returns**

// **Portfolio returns:** [0.12, 0.08, -0.03, 0.15]

// - **Average (mean) return:** (0.12 + 0.08 + (-0.03) + 0.15) / 4 = **0.08**
// - **Excess return:** 0.08 - 0.02 = **0.06**

// **Standard deviation** (using population standard deviation, common for Sharpe ratio calculations): **≈0.0682**

// **Sharpe ratio:** 0.06 / 0.0682 ≈ **0.88**

// *(Note: If using sample standard deviation ≈0.0787, it would be ≈0.76, but population std dev is the typical convention for this metric.)*
        // Uncomment to print the response for inspection
        // for (i, text) in assistant_messages.iter().enumerate() {
        //     println!("Assistant message {}: {}", i, text);
        // }

        // https://docs.x.ai/docs/guides/tools/code-execution-tool#statistical-analysis
        // Statistical Analysis
        // # Hypothesis testing, regression analysis, probability distributions

        let user_prompt = format!(
            "Perform a t-test to compare these two groups and interpret the {}",
            "p-value: Group A: [23, 25, 28, 30], Group B: [20, 22, 24, 26]"
        );
        client.clear_messages();
        client.add_message(Message::user(user_prompt));

        let response = client.send_request_and_parse().unwrap();
        assert!(response.output.len() > 0);

        let assistant_messages = response.extract_assistant_messages();
        assert!(assistant_messages.len() > 0);
        assert_eq!(assistant_messages.len(), 1);

        // Example (actual) response:
        //  **Results of the independent two-sample t-test (assuming equal variances):**

// - **Group A** ([23, 25, 28, 30]): Mean = 26.5, Std = 3.11
// - **Group B** ([20, 22, 24, 26]): Mean = 23.0, Std = 2.58
// - **T-statistic**: 1.73
// - **Degrees of freedom**: 6
// - **P-value**: 0.134

// **Interpretation**:
// The p-value (0.134) is greater than the common significance threshold of 0.05. Therefore, we fail to reject the null hypothesis that the means of the two groups are equal. There is **no statistically significant difference** between Group A and Group B at the 5% significance level. The observed difference in means (3.5) could reasonably occur due to sampling variability alone.

// If you want a Welch's t-test (unequal variances), one-tailed test, or other adjustments, let me know!
        // Uncomment to print the response for inspection
        // for (i, text) in assistant_messages.iter().enumerate() {
        //     println!("Assistant message {}: {}", i, text);
        // }

        // Interestingly, if this assistant message isn't added, the next
        // response will containing responses to the previous user message, as
        // well.
        client.add_message(Message::assistant(assistant_messages[0].clone()));

        // https://docs.x.ai/docs/guides/tools/code-execution-tool#scientific-computing
        // Scientific Computing
        // # Simulations, numerical methods, equation solving
        let user_prompt = format!(
            "Solve this differential equation using numerical methods: {}",
            "dy/dx = x^2 + y, with initial condition y(0) = 1");

        client.add_message(Message::user(user_prompt));

        let response = client.send_request_and_parse().unwrap();
        assert!(response.output.len() > 0);

        let assistant_messages = response.extract_assistant_messages();
        assert!(assistant_messages.len() > 0);
        assert_eq!(assistant_messages.len(), 1);

        // Example (actual) response:
        // ### Numerical Solution of the Differential Equation

// **Equation**: \( \frac{dy}{dx} = x^2 + y \), with initial condition \( y(0) = 1 \).

// **Method**: SciPy's `solve_ivp` (adaptive Runge-Kutta order 4/5, RK45, with relative/absolute tolerances of \( 10^{-8} \)).
// This is a high-order numerical integrator that automatically adjusts step sizes for accuracy and efficiency.

// **Interval**: \( x \in [0, 2] \) (11 evenly spaced points for illustration; solution grows rapidly due to the unstable nature of the equation).

// **Results Table** (numerical approximations, rounded to 6 decimal places):

// | x     | y(x)        |
// |-------|-------------|
// | 0.000 | 1.000000   |
// | 0.200 | 1.224208   |
// | 0.400 | 1.515474   |
// | 0.600 | 1.906356   |
// | 0.800 | 2.436623   |
// | 1.000 | 3.154846   |
// | 1.200 | 4.120351   |
// | 1.400 | 5.405600   |
// | 1.600 | 7.099097   |
// | 1.800 | 9.308942   |
// | 2.000 | 12.167168  |

// **Verification**: The numerical solution matches the exact solution \( y(x) = 3e^x - x^2 - 2x - 2 \) with maximum absolute errors on the order of \( 10^{-8} \), confirming excellent accuracy.

// If you need a different interval, method (e.g., Euler, RK4 manual implementation), more points, or a plot description, let me know!
        // Uncomment to print the response for inspection
        // println!("response.output.len(): {}", response.output.len());
        // for (i, text) in assistant_messages.iter().enumerate() {
        //     println!("Assistant message {}: {}", i, text);
        // }
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