//------------------------------------------------------------------------------
/// Example USAGE:
/// cargo test response_api_clients_tests --test integration_tests
/// cargo test clients::response_api::response_api_clients_test --test integration_tests -- --no-capture
/// Run an individual test:
/// cargo test --test integration_tests test_grok_client_send_request_and_parse -- --nocapture
//------------------------------------------------------------------------------

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

    use common_api::clients::response_api::provider_clients::*;
    use common_api::configurations::model_request_configuration::*;
    use common_api::messages::basic_messages::Message;
    use core_code::utilities::load_environment_file::*;

    #[tokio::test]
    async fn test_grok_client_send_request()
    {
        load_environment_file_from_default_path();
        let configuration = ModelRequestConfiguration::with_model(
            "grok-4-1-fast-reasoning");

        let mut client = GrokResponsesClient::new(
            Some(get_environment_variable_unwrap("XAI_API_KEY")),
            Some(3600),
            Some(configuration)).unwrap();

        let response = client.send_request().await.unwrap();
        
        // Print the response for inspection (executes when test runs)
        // Example (actual) response:
        // Object {
        //     "created_at": Number(1766720065),
        //     "id": String("b64f588b-9f9f-8dc0-4f43-e3055eb4b1a1"),
        //     "incomplete_details": Null,
        //     "max_output_tokens": Null,
        //     "metadata": Object {},
        //     "model": String("grok-4-1-fast-reasoning"),
        //     "object": String("response"),
        //     "output": Array [
        //         Object {
        //             "content": Array [
        //                 Object {
        //                     "annotations": Array [],
        //                     "logprobs": Array [],
        //                     "text": String("#!/bin/bash\n\n#SBATCH --job-name=gromacs_mini\n#SBATCH --output=gromacs_mini_%J.out\n#SBATCH --error=gromacs_mini_%J.err\n#SBATCH --ntasks=1\n#SBATCH --mem=32G\n#SBATCH --cpus-per-task=16\n#SBATCH --gres=gres:psi:1\n#SBATCH --partition=plgrid-testing\n#SBATCH --time=00:30:00\n\n# Load modules (load the CPU version of the module)\nmodule add gromacs/2022.5-CPU\n\n# Set Gromacs to use as many cores as allocated\nexport OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n\n# Input gro/pp/ndw/top files\ngrofile=gromacs_mini.nostop.gro\nt"),
        //                     "type": String("output_text"),
        //                 },
        //             ],
        //             "id": String("msg_b64f588b-9f9f-8dc0-4f43-e3055eb4b1a1"),
        //             "role": String("assistant"),
        //             "status": String("completed"),
        //             "type": String("message"),
        //         },
        //     ],
        //     "parallel_tool_calls": Bool(true),
        //     "previous_response_id": Null,
        //     "reasoning": Object {
        //         "effort": String("medium"),
        //         "summary": String("detailed"),
        //     },
        //     "status": String("completed"),
        //     "store": Bool(true),
        //     "temperature": Null,
        //     "text": Object {
        //         "format": Object {
        //             "type": String("text"),
        //         },
        //     },
        //     "tool_choice": String("auto"),
        //     "tools": Array [],
        //     "top_p": Null,
        //     "usage": Object {
        //         "input_tokens": Number(2),
        //         "input_tokens_details": Object {
        //             "cached_tokens": Number(0),
        //         },
        //         "num_server_side_tools_used": Number(0),
        //         "num_sources_used": Number(0),
        //         "output_tokens": Number(381),
        //         "output_tokens_details": Object {
        //             "reasoning_tokens": Number(224),
        //         },
        //         "total_tokens": Number(383),
        //     },
        //     "user": Null,
        // }

        // {
        //     "created_at": 1766720065,
        //     "id": "b64f588b-9f9f-8dc0-4f43-e3055eb4b1a1",
        //     "incomplete_details": null,
        //     "max_output_tokens": null,
        //     "metadata": {},
        //     "model": "grok-4-1-fast-reasoning",
        //     "object": "response",
        //     "output": [
        //       {
        //         "content": [
        //           {
        //             "annotations": [],
        //             "logprobs": [],
        //             "text": "#!/bin/bash\n\n#SBATCH --job-name=gromacs_mini\n#SBATCH --output=gromacs_mini_%J.out\n#SBATCH --error=gromacs_mini_%J.err\n#SBATCH --ntasks=1\n#SBATCH --mem=32G\n#SBATCH --cpus-per-task=16\n#SBATCH --gres=gres:psi:1\n#SBATCH --partition=plgrid-testing\n#SBATCH --time=00:30:00\n\n# Load modules (load the CPU version of the module)\nmodule add gromacs/2022.5-CPU\n\n# Set Gromacs to use as many cores as allocated\nexport OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n\n# Input gro/pp/ndw/top files\ngrofile=gromacs_mini.nostop.gro\nt",
        //             "type": "output_text"
        //           }
        //         ],
        //         "id": "msg_b64f588b-9f9f-8dc0-4f43-e3055eb4b1a1",
        //         "role": "assistant",
        //         "status": "completed",
        //         "type": "message"
        //       }
        //     ],
        //     "parallel_tool_calls": true,
        //     "previous_response_id": null,
        //     "reasoning": {
        //       "effort": "medium",
        //       "summary": "detailed"
        //     },
        //     "status": "completed",
        //     "store": true,
        //     "temperature": null,
        //     "text": {
        //       "format": {
        //         "type": "text"
        //       }
        //     },
        //     "tool_choice": "auto",
        //     "tools": [],
        //     "top_p": null,
        //     "usage": {
        //       "input_tokens": 2,
        //       "input_tokens_details": {
        //         "cached_tokens": 0
        //       },
        //       "num_server_side_tools_used": 0,
        //       "num_sources_used": 0,
        //       "output_tokens": 381,
        //       "output_tokens_details": {
        //         "reasoning_tokens": 224
        //       },
        //       "total_tokens": 383
        //     },
        //     "user": null
        //   }

        // Uncomment to print the response for inspection
        // println!("API Response (debug): {:#?}", response);
        // if let Ok(pretty) = serde_json::to_string_pretty(&response) {
        //     println!("Pretty JSON:\n{}", pretty);
        // } else {
        //     println!("Raw string response: {}", response);
        // }
        // Basic assert to use response (avoid unused warning)
        assert!(
            response.is_object() || response.is_null(),
            "Unexpected response format: {:?}",
            response);

        assert_eq!(client.client().responses_client.messages().len(), 0);

        client.add_message(Message::system(
            "You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy."));

        client.add_message(Message::user(
            "What is the meaning of life, the universe, and everything?"));

        assert_eq!(client.client().responses_client.messages().len(), 2);

        let response = client.send_request().await.unwrap();

        // Example (actual) response:
        // Object {
        //     "created_at": Number(1766720069),
        //     "id": String("cbf1dee3-cd9a-cb7d-f5e3-341556d2be81"),
        //     "incomplete_details": Null,
        //     "max_output_tokens": Null,
        //     "metadata": Object {},
        //     "model": String("grok-4-1-fast-reasoning"),
        //     "object": String("response"),
        //     "output": Array [
        //         Object {
        //             "content": Array [
        //                 Object {
        //                     "annotations": Array [],
        //                     "logprobs": Array [],
        //                     "text": String("42"),
        //                     "type": String("output_text"),
        //                 },
        //             ],
        //             "id": String("msg_cbf1dee3-cd9a-cb7d-f5e3-341556d2be81"),
        //             "role": String("assistant"),
        //             "status": String("completed"),
        //             "type": String("message"),
        //         },
        //     ],
        //     "parallel_tool_calls": Bool(true),
        //     "previous_response_id": Null,
        //     "reasoning": Object {
        //         "effort": String("medium"),
        //         "summary": String("detailed"),
        //     },
        //     "status": String("completed"),
        //     "store": Bool(true),
        //     "temperature": Null,
        //     "text": Object {
        //         "format": Object {
        //             "type": String("text"),
        //         },
        //     },
        //     "tool_choice": String("auto"),
        //     "tools": Array [],
        //     "top_p": Null,
        //     "usage": Object {
        //         "input_tokens": Number(187),
        //         "input_tokens_details": Object {
        //             "cached_tokens": Number(149),
        //         },
        //         "num_server_side_tools_used": Number(0),
        //         "num_sources_used": Number(0),
        //         "output_tokens": Number(152),
        //         "output_tokens_details": Object {
        //             "reasoning_tokens": Number(151),
        //         },
        //         "total_tokens": Number(339),
        //     },
        //     "user": Null,
        // }

        // {
        //     "created_at": 1766720069,
        //     "id": "cbf1dee3-cd9a-cb7d-f5e3-341556d2be81",
        //     "incomplete_details": null,
        //     "max_output_tokens": null,
        //     "metadata": {},
        //     "model": "grok-4-1-fast-reasoning",
        //     "object": "response",
        //     "output": [
        //       {
        //         "content": [
        //           {
        //             "annotations": [],
        //             "logprobs": [],
        //             "text": "42",
        //             "type": "output_text"
        //           }
        //         ],
        //         "id": "msg_cbf1dee3-cd9a-cb7d-f5e3-341556d2be81",
        //         "role": "assistant",
        //         "status": "completed",
        //         "type": "message"
        //       }
        //     ],
        //     "parallel_tool_calls": true,
        //     "previous_response_id": null,
        //     "reasoning": {
        //       "effort": "medium",
        //       "summary": "detailed"
        //     },
        //     "status": "completed",
        //     "store": true,
        //     "temperature": null,
        //     "text": {
        //       "format": {
        //         "type": "text"
        //       }
        //     },
        //     "tool_choice": "auto",
        //     "tools": [],
        //     "top_p": null,
        //     "usage": {
        //       "input_tokens": 187,
        //       "input_tokens_details": {
        //         "cached_tokens": 149
        //       },
        //       "num_server_side_tools_used": 0,
        //       "num_sources_used": 0,
        //       "output_tokens": 152,
        //       "output_tokens_details": {
        //         "reasoning_tokens": 151
        //       },
        //       "total_tokens": 339
        //     },
        //     "user": null
        //   }

        // Uncomment to print the response for inspection
        // println!("API Response (debug): {:#?}", response);
        // if let Ok(pretty) = serde_json::to_string_pretty(&response) {
        //     println!("Pretty JSON:\n{}", pretty);
        // } else {
        //     println!("Raw string response: {}", response);
        // }
        assert_eq!(client.client().responses_client.messages().len(), 2);
        assert!(
            response.is_object() || response.is_null(),
            "Unexpected response format: {:?}",
            response);
    }

    #[tokio::test]
    async fn test_grok_client_send_request_and_parse()
    {
        load_environment_file_from_default_path();
        let configuration = ModelRequestConfiguration::with_model(
            "grok-4-1-fast-reasoning");

        let mut client = GrokResponsesClient::new(
            Some(get_environment_variable_unwrap("XAI_API_KEY")),
            Some(3600),
            Some(configuration)).unwrap();

        client.add_message(Message::system(
            "You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy."));

        // https://platform.openai.com/docs/api-reference/responses#:~:text=Tell%20me%20a%20three%20sentence%20bedtime%20story%20about%20a%20unicorn%2E%22
        client.add_message(Message::user(
            "Tell me a three sentence bedtime story about a unicorn."));

        assert_eq!(client.client().responses_client.messages().len(), 2);
       
        let _response = client.send_request_and_parse().await.unwrap();

        // Example (actual) response:
        // 4614e015-5277-0fc1-f64f-df91d5a29f4a
        // Uncomment to print the response for inspection
        // println!("_response.id: {}", _response.id);

        // Example (actual) response:
        // In a enchanted forest bathed in moonlight, a gentle unicorn named Luna wandered through meadows of silver grass, her horn glowing like a fallen star.  
        // She discovered a hidden stream where fireflies danced and whispered secrets of the night, filling her heart with wonder and peace.  
        // As the stars twinkled goodnight, Luna curled up on a bed of soft petals, drifting into sweet dreams until the dawn.
        // Uncomment to print the response for inspection
        // println!(
        //     "_response.output[0].content[0].text: {}",
        //     _response.output[0].content[0].text);
    }

    //--------------------------------------------------------------------------
    /// https://console.groq.com/docs/responses-api
    //--------------------------------------------------------------------------

    #[test]
    fn test_grok_blocking_client_send_request()
    {
        load_environment_file_from_default_path();
        let configuration = ModelRequestConfiguration::with_model(
            "llama-3.3-70b-versatile");

        let mut client = GroqResponsesBlockingClient::new(
            Some(get_environment_variable_unwrap("GROQ_API_KEY")),
            None,
            Some(configuration)).unwrap();

        client.add_message(Message::user(
            "Tell me a fun fact about the moon in one sentence."
        ));

        let _request_value =
        client.client().responses_client.input_data().build_request_value().unwrap();
        println!(
            "request value: {}",
            _request_value,
        );

        let _response = client.send_request().unwrap();

        // Example (actual) response:

        // API Response (debug): Object {
        //     "background": Bool(false),
        //     "created_at": Number(1766899931),
        //     "error": Null,
        //     "groq": Null,
        //     "id": String("resp_01kdhq2nw6factarzy30tt5b2r"),
        //     "incomplete_details": Null,
        //     "max_output_tokens": Null,
        //     "max_tool_calls": Null,
        //     "metadata": Object {},
        //     "model": String("llama-3.3-70b-versatile"),
        //     "object": String("response"),
        //     "output": Array [
        //         Object {
        //             "id": String("resp_01kdhq2nw6fad9td3y1trasax6"),
        //             "status": String("completed"),
        //             "summary": Array [],
        //             "type": String("reasoning"),
        //         },
        //         Object {
        //             "content": Array [
        //                 Object {
        //                     "annotations": Array [],
        //                     "logprobs": Null,
        //                     "text": String("The moon is not round in shape, but is actually an ellipsoid, meaning it's slightly flattened at the poles and bulging at the equator, due to its rotation and the gravitational pull of the Earth."),
        //                     "type": String("output_text"),
        //                 },
        //             ],
        //             "id": String("msg_01kdhq2nw6fadvrd9wcs1xcytn"),
        //             "role": String("assistant"),
        //             "status": String("completed"),
        //             "type": String("message"),
        //         },
        //     ],
        //     "parallel_tool_calls": Bool(true),
        //     "previous_response_id": Null,
        //     "reasoning": Null,
        //     "service_tier": String("default"),
        //     "status": String("completed"),
        //     "store": Bool(false),
        //     "temperature": Number(1),
        //     "text": Object {
        //         "format": Object {
        //             "type": String("text"),
        //         },
        //     },
        //     "tool_choice": String("auto"),
        //     "tools": Array [],
        //     "top_logprobs": Null,
        //     "top_p": Number(1),
        //     "truncation": String("disabled"),
        //     "usage": Object {
        //         "input_tokens": Number(47),
        //         "input_tokens_details": Object {
        //             "cached_tokens": Number(0),
        //         },
        //         "output_tokens": Number(44),
        //         "output_tokens_details": Object {
        //             "reasoning_tokens": Number(0),
        //         },
        //         "total_tokens": Number(91),
        //     },
        //     "user": Null,
        // }
        // Pretty JSON:
        // {
        //   "background": false,
        //   "created_at": 1766899931,
        //   "error": null,
        //   "groq": null,
        //   "id": "resp_01kdhq2nw6factarzy30tt5b2r",
        //   "incomplete_details": null,
        //   "max_output_tokens": null,
        //   "max_tool_calls": null,
        //   "metadata": {},
        //   "model": "llama-3.3-70b-versatile",
        //   "object": "response",
        //   "output": [
        //     {
        //       "id": "resp_01kdhq2nw6fad9td3y1trasax6",
        //       "status": "completed",
        //       "summary": [],
        //       "type": "reasoning"
        //     },
        //     {
        //       "content": [
        //         {
        //           "annotations": [],
        //           "logprobs": null,
        //           "text": "The moon is not round in shape, but is actually an ellipsoid, meaning it's slightly flattened at the poles and bulging at the equator, due to its rotation and the gravitational pull of the Earth.",
        //           "type": "output_text"
        //         }
        //       ],
        //       "id": "msg_01kdhq2nw6fadvrd9wcs1xcytn",
        //       "role": "assistant",
        //       "status": "completed",
        //       "type": "message"
        //     }
        //   ],
        //   "parallel_tool_calls": true,
        //   "previous_response_id": null,
        //   "reasoning": null,
        //   "service_tier": "default",
        //   "status": "completed",
        //   "store": false,
        //   "temperature": 1,
        //   "text": {
        //     "format": {
        //       "type": "text"
        //     }
        //   },
        //   "tool_choice": "auto",
        //   "tools": [],
        //   "top_logprobs": null,
        //   "top_p": 1,
        //   "truncation": "disabled",
        //   "usage": {
        //     "input_tokens": 47,
        //     "input_tokens_details": {
        //       "cached_tokens": 0
        //     },
        //     "output_tokens": 44,
        //     "output_tokens_details": {
        //       "reasoning_tokens": 0
        //     },
        //     "total_tokens": 91
        //   },
        //   "user": null
        // }

        // Uncomment to print the response for inspection
        // println!("API Response (debug): {:#?}", _response);
        // if let Ok(pretty) = serde_json::to_string_pretty(&_response) {
        //     println!("Pretty JSON:\n{}", pretty);
        // } else {
        //     println!("Raw string response: {}", _response);
        // }
    }

    #[test]
    fn test_groq_blocking_client_send_request_and_parse()
    {
        load_environment_file_from_default_path();
        let configuration = ModelRequestConfiguration::with_model(
            "openai/gpt-oss-20b");

        let mut client = GroqResponsesBlockingClient::new(
            Some(get_environment_variable_unwrap("GROQ_API_KEY")),
            None,
            Some(configuration)).unwrap();

        client.add_message(Message::user(
            "Tell me a fun fact about the moon in one sentence."
        ));

        // Example (actual) request value:
        // request value: {"input":[{"content":"Tell me a fun fact about the moon in one sentence.","role":"user"}],"model":"openai/gpt-oss-20b"}

        // Uncomment to print the request value for inspection
        // let _request_value =
        //     client.client().responses_client.input_data().build_request_value().unwrap();
        // println!(
        //     "request value: {}",
        //     _request_value,
        // );

        let _response = client.send_request_and_parse().unwrap();

        assert!(_response.output.len() > 0);
        assert_eq!(_response.output.len(), 2);

        assert_eq!(_response.output[1].type_, "message");
        assert_eq!(_response.output[1].role, "assistant");
        assert_eq!(_response.output[1].content[0].type_, "output_text");

        // Example (actual) response:
        // _response.output[1].content.text: Scientists have discovered that pockets of water ice hidden in the Moon’s permanently shadowed polar craters could one day supply future astronauts with drinking water and rocket fuel.
        // Uncomment to print the response for inspection
        // println!(
        //     "_response.output[1].content.text: {}",
        //     _response.output[1].content[0].text);
    }

    #[test]
    fn test_groq_blocking_client_for_multi_turn_conversations()
    {
        load_environment_file_from_default_path();
        let configuration = ModelRequestConfiguration::with_model(
            "openai/gpt-oss-20b");

        let mut client = GroqResponsesBlockingClient::new(
            Some(get_environment_variable_unwrap("GROQ_API_KEY")),
            None,
            Some(configuration)).unwrap();

        // https://github.com/microsoft/lost_in_conversation/blob/main/prompts/math/gsm8k_lazification_conversational.txt
        let user_message_content = format!(
            "There are 15 trees in the grove. Grove workers will plant {}{}",
            "trees in the grove today. After they are done, there will be 21 ",
            "trees. How many trees did the grove workers plant today?");

        client.add_message(Message::user(user_message_content.clone()));

        let response = client.send_request_and_parse().unwrap();

        let is_assistant_message = response.output[1].role == "assistant";

        if is_assistant_message
        {
            let assistant_message_content =
                response.output[1].content[0].text.clone();
            client.add_message(Message::assistant(assistant_message_content));
        }

        // https://github.com/microsoft/lost_in_conversation/blob/main/prompts/math/math_system_prompt.txt
        let system_message_content = format!(
            "As an expert problem solver solve step by step the following {}",
            "mathematical questions.");
        client.add_message(Message::system(system_message_content));

        client.add_message(Message::user(user_message_content.clone()));

        let response = client.send_request_and_parse().unwrap();

        let is_assistant_message = response.output[1].role == "assistant";
        if is_assistant_message
        {
            let assistant_message_content =
                response.output[1].content[0].text.clone();
            client.add_message(Message::assistant(assistant_message_content));
        }

        // https://github.com/microsoft/lost_in_conversation/blob/main/prompts/math/gsm8k_lazification_segment.txt
        let user_message_content =
            "How many trees did the grove workers plant today?";
        client.add_message(Message::user(user_message_content));

        let response = client.send_request_and_parse().unwrap();

        let is_assistant_message = response.output[1].role == "assistant";
        if is_assistant_message
        {
            let assistant_message_content =
                response.output[1].content[0].text.clone();
            client.add_message(Message::assistant(assistant_message_content));
        }

        assert_eq!(client.client().responses_client.messages().len(), 7);

        // Example (actual) messages:
        // index: 0, message: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
        // message json: {
        //   "role": "user",
        //   "content": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"
        // }
        // index: 1, message: The workers planted **6 trees** today.

        // Starting with 15 trees, adding 6 gives \(15 + 6 = 21\) trees.
        // message json: {
        //   "role": "assistant",
        //   "content": "The workers planted **6 trees** today.  \n\nStarting with 15 trees, adding 6 gives \\(15 + 6 = 21\\) trees."
        // }
        // index: 2, message: As an expert problem solver solve step by step the following mathematical questions.
        // message json: {
        //   "role": "system",
        //   "content": "As an expert problem solver solve step by step the following mathematical questions."
        // }
        // index: 3, message: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
        // message json: {
        //   "role": "user",
        //   "content": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"
        // }
        // index: 4, message: The workers planted \(6\) trees.

        // **Why?**
        // - Initially there are \(15\) trees.
        // - After planting, there are \(21\) trees.
        // - The number of trees added is \(21 - 15 = 6\).

        // So the grove workers planted **six** trees today.
        // message json: {
        //   "role": "assistant",
        //   "content": "The workers planted \\(6\\) trees.\n\n**Why?**  \n- Initially there are \\(15\\) trees.  \n- After planting, there are \\(21\\) trees.  \n- The number of trees added is \\(21 - 15 = 6\\).\n\nSo the grove workers planted **six** trees today."
        // }
        // index: 5, message: How many trees did the grove workers plant today?
        // message json: {
        //   "role": "user",
        //   "content": "How many trees did the grove workers plant today?"
        // }
        // index: 6, message: The grove workers planted **6 trees** today.

        // - **Start**: 15 trees
        // - **After planting**: 21 trees
        // - **Added**: \(21 - 15 = 6\)

        // So 6 trees were planted.
        // message json: {
        //   "role": "assistant",
        //   "content": "The grove workers planted **6 trees** today.\n\n- **Start**: 15 trees  \n- **After planting**: 21 trees  \n- **Added**: \\(21 - 15 = 6\\)\n\nSo 6 trees were planted."
        // }

        // Uncomment to print the messages for inspection
        // for (index, message)
        //     in client.client().responses_client.messages().iter().enumerate()
        // {
        //     println!("index: {}, message: {}", index, message.content);
        //     println!("message json: {}", message.to_json_pretty().unwrap());
        // }
    }
}