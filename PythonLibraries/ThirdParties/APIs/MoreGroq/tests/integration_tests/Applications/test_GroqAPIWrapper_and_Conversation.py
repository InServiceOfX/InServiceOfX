from commonapi.Messages import (
    AssistantMessage,
    ConversationAndSystemMessages,
    ParsePromptsCollection,
    UserMessage)
from corecode.FileIO import JSONFile, TextFile
from corecode.SetupProjectData.SetupPrompts import \
    ParseJujumilk3LeakedSystemPrompts
from corecode.Utilities import (
    DataSubdirectories,
    get_environment_variable,
    load_environment_file,
    setup_datasets_path)
from moregroq.Wrappers import GroqAPIWrapper
from moregroq.Wrappers.ChatCompletionConfiguration \
    import ChatCompletionConfiguration
from moretransformers.Applications.Datasets import ParseOpenAssistantOasst1
from moretransformers.Wrappers.Datasets import LoadAndSaveLocally
from pathlib import Path

import pytest

load_environment_file()

data_subdirectories = DataSubdirectories()

def path_for_system_prompts_0():
    return ParseJujumilk3LeakedSystemPrompts()._repo_path

def path_for_example_dataset_0():
    datasets_path = setup_datasets_path()
    return datasets_path / "OpenAssistant" / "oasst1"

def path_for_prompts_collection():
    return data_subdirectories.PromptsCollection

@pytest.fixture
def train_prompts_and_system_prompt():
    """Fixture that provides train_prompts and system_prompt for tests."""
    datasets_path = setup_datasets_path()
    load_and_save_locally = LoadAndSaveLocally(datasets_path)
    dataset_name = "OpenAssistant/oasst1"
    dataset = load_and_save_locally.load_from_disk(dataset_name)
    train_prompts = ParseOpenAssistantOasst1.parse_for_train_prompter(dataset)
    train_english_prompts = \
        ParseOpenAssistantOasst1.parse_for_train_prompter_english(dataset)

    parse_jujumilk3_leaked_system_prompts = ParseJujumilk3LeakedSystemPrompts()
    system_prompt, _, _ = \
        parse_jujumilk3_leaked_system_prompts.get_anthropic_claude_3_7_sonnet_prompt()
    
    return train_prompts, train_english_prompts, system_prompt

@pytest.fixture
def train_prompts_and_system_prompt_1():
    """Fixture that provides train_prompts and system_prompt for tests."""
    datasets_path = setup_datasets_path()
    load_and_save_locally = LoadAndSaveLocally(datasets_path)
    dataset_name = "OpenAssistant/oasst1"
    dataset = load_and_save_locally.load_from_disk(dataset_name)
    train_prompts = ParseOpenAssistantOasst1.parse_for_train_prompter(dataset)
    train_english_prompts = \
        ParseOpenAssistantOasst1.parse_for_train_prompter_english(dataset)

    parse_jujumilk3_leaked_system_prompts = ParseJujumilk3LeakedSystemPrompts()
    system_prompt_template = \
        parse_jujumilk3_leaked_system_prompts.get_meta_ai_whatsapp_string_template()

    system_prompt = system_prompt_template.format_prompt(
        model_name="Meta AI",
        model_engine="Llama 4",
        company="Meta",
        date="2025-07-08",
        location="Paris"
    )

    return train_prompts, train_english_prompts, system_prompt

@pytest.mark.skipif(
    not path_for_system_prompts_0().exists() or \
        not path_for_example_dataset_0().exists(),
    reason=(
        "Repository jujumilk3/leaked-system-prompts or OpenAssistant/oasst1 "
        "not found locally"))
def test_GroqAPIWrapper_and_ConversationAndSystemMessages_works(
    train_prompts_and_system_prompt):
    _, train_english_prompts, system_prompt = \
        train_prompts_and_system_prompt

    chat_completion_configuration = ChatCompletionConfiguration()
    chat_completion_configuration.model = "deepseek-r1-distill-llama-70b"

    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.configuration = chat_completion_configuration

    conversation_and_system_messages = ConversationAndSystemMessages()
    conversation_and_system_messages.add_system_message(system_prompt)

    model_responses = []
    model_response_indices = []

    # E               groq.RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01hxms56tvftrr5ysv46mybzqc` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 92762, Requested 12452. Please try again in 1h15m4.753s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}

#/usr/local/lib/python3.12/dist-packages/groq/_base_client.py:1034: RateLimitError

    try:
        for index, prompt in enumerate(train_english_prompts):
            conversation_and_system_messages.append_message(
                UserMessage(content=prompt["text"]))

            conversation_as_input = \
                conversation_and_system_messages.get_conversation_as_list_of_dicts()

            response = groq_api_wrapper.create_chat_completion(
                conversation_as_input)
            if response and hasattr(response, "choices") and \
                len(response.choices) > 0:
                assistant_message_content = response.choices[0].message.content
                conversation_and_system_messages.append_message(
                    AssistantMessage(content=assistant_message_content))
                # Uncomment to print the assistant message content
                print(assistant_message_content)
                model_responses.append(response.choices[0].message)
            else:
                print(f"No response received from API call: {response}")
                conversation_and_system_messages.append_general_message(response)

            model_responses.append(response)
            model_response_indices.append(index)
    except Exception as e:
        print(f"Error: {e}")
        print(model_responses)
        if len(model_response_indices) > 0:
            print("model_response_indices[-1]", model_response_indices[-1])

            model_responses_path = Path.cwd() / "model_responses.json"
            JSONFile.save_json(model_responses_path, model_responses)
            conversation_path = Path.cwd() / "conversation.json"
            JSONFile.save_json(
                conversation_path,
                conversation_and_system_messages.get_conversation_as_list_of_dicts())

        else:
            print("model_response_indices is empty")
        raise

    assert len(model_responses) == len(train_prompts)

    model_responses_path = Path.cwd() / "model_responses.json"
    JSONFile.save_json(model_responses_path, model_responses)
    conversation_path = Path.cwd() / "conversation.json"
    JSONFile.save_json(
        conversation_path,
        conversation_and_system_messages.get_conversation_as_list_of_dicts())

    print("\n\n\n Done with model responses")

    for model_response in model_responses:
        try:
            print(model_response.choices[0].message.content)
        except Exception as e:
            print(f"Error: {e}")
            print(model_response)


@pytest.mark.skipif(
    not path_for_system_prompts_0().exists() or \
        not path_for_example_dataset_0().exists(),
    reason=(
        "Repository jujumilk3/leaked-system-prompts or OpenAssistant/oasst1 "
        "not found locally"))
def test_GroqAPIWrapper_and_ConversationAndSystemMessages_works_on_a_few_prompts(
    train_prompts_and_system_prompt_1):
    _, train_english_prompts, system_prompt = \
        train_prompts_and_system_prompt_1

    chat_completion_configuration = ChatCompletionConfiguration()
    chat_completion_configuration.model = \
        "meta-llama/llama-4-scout-17b-16e-instruct"
    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.configuration = chat_completion_configuration

    conversation_and_system_messages = ConversationAndSystemMessages()
    conversation_and_system_messages.add_system_message(system_prompt)

    model_responses = []
    model_response_indices = []

    number_of_prompts = 1

    try:
        for index, prompt in enumerate(
            train_english_prompts[:number_of_prompts]):
            conversation_and_system_messages.append_message(
                UserMessage(content=prompt["text"]))

            conversation_as_input = \
                conversation_and_system_messages.get_conversation_as_list_of_dicts()

            response = groq_api_wrapper.create_chat_completion(
                conversation_as_input)
            if response and hasattr(response, "choices") and \
                len(response.choices) > 0:
                assistant_message_content = response.choices[0].message.content
                conversation_and_system_messages.append_message(
                    AssistantMessage(content=assistant_message_content))
                # Uncomment to print the assistant message content
                print(assistant_message_content)
                
                # Comment and uncomment out lines of code to save and see what
                # different response types the Groq API returns.
                #model_responses.append(response.choices[0].message)
                model_responses.append(response)
            else:
                print(f"No response message received from API call: {response}")
                conversation_and_system_messages.append_general_message(response)
                model_responses.append(response)

            model_response_indices.append(index)
    except Exception as e:
        print(f"Error: {e}")
        print(model_responses)
        if len(model_response_indices) > 0:
            print("model_response_indices[-1]", model_response_indices[-1])

            conversation_path = Path.cwd() / "conversation.json"
            JSONFile.save_json(
                conversation_path,
                conversation_and_system_messages.get_conversation_as_list_of_dicts())

            model_responses_path = Path.cwd() / "model_responses.json"
            JSONFile.save_json(model_responses_path, model_responses)

        else:
            print("model_response_indices is empty")
        raise

    assert len(model_responses) == number_of_prompts

    conversation_path = Path.cwd() / "conversation.json"
    JSONFile.save_json(
        conversation_path,
        conversation_and_system_messages.get_conversation_as_list_of_dicts())

    # <class 'groq.types.chat.chat_completion.ChatCompletion'>
    # print("type(model_responses[0])", type(model_responses[0]))

    # for response
    # ChatCompletion(id='chatcmpl-660a9ce1-bfa3-4320-b009-531de9c361b4', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Monopsony and monopoly are related but distinct concepts. A monopoly occurs when a single company dominates the supply side of a market, while a monopsony dominates the demand side.\n\nHistorically, monopsony power has emerged in various industries and economies. Here are a few examples:\n\n1. **Pre-WWII US Steel Industry**: The steel industry in the United States was dominated by a few large firms, including U.S. Steel, which controlled over 50% of the market. This concentration of market power allowed these firms to dictate wages and working conditions.\n\n2. **Post-WWII US Coal Industry**: The coal industry in the United States experienced significant consolidation, leading to a small number of large firms dominating the market. This concentration of power allowed these firms to exert significant influence over wages and working conditions.\n\n3. **Modern-Day Tech Industry**: The tech industry, particularly in areas like Silicon Valley, has seen significant consolidation, with a few large firms like Google, Amazon, and Facebook dominating the market. This has led to concerns about monopsony power and its impact on workers.\n\nCommon prerequisites for monopsony power to emerge include:\n\n* **Market concentration**: A decrease in the number of firms operating in a market can lead to increased concentration and monopsony power.\n\n* **Barriers to entry**: High barriers to entry, such as significant startup costs or regulatory hurdles, can prevent new firms from entering the market and challenging existing firms.\n\n* **Lack of worker mobility**: When workers have limited ability to move between firms or industries, they are more vulnerable to monopsony power.\n\n* **Weak labor laws and regulations**: Inadequate labor laws and regulations can allow firms to exploit workers and maintain monopsony power.\n\n* **Globalization and technological change**: Shifts in global trade patterns and technological advancements can lead to industry consolidation and increased monopsony power.\n\nThese factors can create an environment where monopsony power can emerge and persist.', role='assistant', executed_tools=None, function_call=None, reasoning=None, tool_calls=None))], created=1751977661, model='meta-llama/llama-4-scout-17b-16e-instruct', object='chat.completion', system_fingerprint='fp_79da0e0073', usage=CompletionUsage(completion_tokens=387, prompt_tokens=2436, total_tokens=2823, completion_time=0.783724838, prompt_time=0.073756772, queue_time=0.211007971, total_time=0.85748161), usage_breakdown=None, x_groq={'id': 'req_01jzn034etfkva86zyj3y8x3kg'})

    # for response.choices[0].message
    # ChatCompletionMessage(content="A monopsony and a monopoly are similar in that both refer to a market structure where one entity has significant control. However, a monopoly refers to a single seller dominating a market, while a monopsony refers to a single buyer dominating a market.\n\nHere are some historical examples of monopsony-like structures consolidating over time:\n\n### Historical Examples \n\n1. **The company town phenomenon**: In the late 19th and early 20th centuries, many company towns in the United States were dominated by a single employer, such as a mining or manufacturing company. The company controlled not only the jobs but also the housing, stores, and other services, creating a monopsony-like situation.\n2. **The coal mining industry in Appalachia**: During the early 20th century, coal mining companies in Appalachia dominated the labor market, controlling wages, working conditions, and even the local economy. This created a monopsony-like situation, where workers had limited job opportunities and were subject to exploitation.\n3. **The farmworker situation in California**: In the mid-20th century, large agricultural companies in California, such as the growers' associations, dominated the labor market for farmworkers. This led to low wages, poor working conditions, and limited job opportunities for workers.\n\n### Common Prerequisites \n\nSome common circumstances that can lead to a monopsony-like structure include:\n\n* **Economic concentration**: When a few large companies dominate an industry, it can create a monopsony-like situation.\n* **Limited job opportunities**: When there are few job opportunities in a particular region or industry, workers may be more vulnerable to exploitation by a single employer.\n* **Lack of worker mobility**: When workers have limited ability to move to other regions or industries, they may be more susceptible to a monopsonistic employer.\n* **Weak labor unions**: When labor unions are weak or absent, workers may have limited bargaining power to negotiate better wages and working conditions.\n* **Government policies**: Government policies, such as lax antitrust enforcement or weak labor laws, can contribute to the creation of a monopsony-like structure.\n\n### Examples from Specific Industries \n\n* **The steel industry in the United States**: In the late 19th and early 20th centuries, a few large steel companies, such as U.S. Steel, dominated the industry and exerted significant control over workers.\n* **The technology industry in Silicon Valley**: Some argue that large tech companies, such as Google and Facebook, have created a monopsony-like situation in the tech industry, with limited job opportunities and high barriers to entry for new companies.\n\nThese examples illustrate how monopsony-like structures can emerge in different industries and contexts, often as a result of a combination of economic, social, and policy factors.", role='assistant', executed_tools=None, function_call=None, reasoning=None, tool_calls=None)

    for model_response in model_responses:
        print("model_response", model_response)

    # model_responses_path = Path.cwd() / "model_responses.json"
    # JSONFile.save_json(model_responses_path, model_responses)

def test_works_with_smaller_max_completion_tokens(
    train_prompts_and_system_prompt_1):
    _, train_english_prompts, system_prompt = \
        train_prompts_and_system_prompt_1

    chat_completion_configuration = ChatCompletionConfiguration()
    chat_completion_configuration.model = \
        "llama-3.1-8b-instant"
#        "meta-llama/llama-4-scout-17b-16e-instruct"
    # I tried these values that I commented out as well, but I was aiming for
    # decreasing the length of answers in a meaningful way.
    #chat_completion_configuration.max_completion_tokens = 1024
    #chat_completion_configuration.max_completion_tokens = 512
    chat_completion_configuration.max_completion_tokens = 256

    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.configuration = chat_completion_configuration

    conversation_and_system_messages = ConversationAndSystemMessages()
    conversation_and_system_messages.add_system_message(system_prompt)

    model_responses = []

    number_of_prompts = 3

    try:
        for prompt in train_english_prompts[:number_of_prompts]:
            conversation_and_system_messages.append_message(
                UserMessage(content=prompt["text"]))

            conversation_as_input = \
                conversation_and_system_messages.get_conversation_as_list_of_dicts()

            response = groq_api_wrapper.create_chat_completion(
                conversation_as_input)
            if response and hasattr(response, "choices") and \
                len(response.choices) > 0:
                assistant_message_content = response.choices[0].message.content
                conversation_and_system_messages.append_message(
                    AssistantMessage(content=assistant_message_content))
                # Uncomment to print the assistant message content
                print(assistant_message_content)
                
                # Comment and uncomment out lines of code to save and see what
                # different response types the Groq API returns.
                #model_responses.append(response.choices[0].message)
                model_responses.append(response.choices[0].message)
            else:
                print(f"No response message received from API call: {response}")
                conversation_and_system_messages.append_general_message(response)

    except Exception as e:
        print(f"Error: {e}")
        print(model_responses)
        if len(model_responses) > 0:

            conversation_path = Path.cwd() / "conversation.json"
            JSONFile.save_json(
                conversation_path,
                conversation_and_system_messages.get_conversation_as_list_of_dicts())

            model_responses_path = Path.cwd() / "model_responses.json"
            JSONFile.save_json(model_responses_path, model_responses)
        else:
            print("model_responses is empty")
        raise

    assert len(model_responses) == number_of_prompts

    conversation_path = Path.cwd() / "conversation.json"
    JSONFile.save_json(
        conversation_path,
        conversation_and_system_messages.get_conversation_as_list_of_dicts())

    # <class 'groq.types.chat.chat_completion.ChatCompletion'>
    # print("type(model_responses[0])", type(model_responses[0]))

    # for response
    # ChatCompletion(id='chatcmpl-660a9ce1-bfa3-4320-b009-531de9c361b4', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Monopsony and monopoly are related but distinct concepts. A monopoly occurs when a single company dominates the supply side of a market, while a monopsony dominates the demand side.\n\nHistorically, monopsony power has emerged in various industries and economies. Here are a few examples:\n\n1. **Pre-WWII US Steel Industry**: The steel industry in the United States was dominated by a few large firms, including U.S. Steel, which controlled over 50% of the market. This concentration of market power allowed these firms to dictate wages and working conditions.\n\n2. **Post-WWII US Coal Industry**: The coal industry in the United States experienced significant consolidation, leading to a small number of large firms dominating the market. This concentration of power allowed these firms to exert significant influence over wages and working conditions.\n\n3. **Modern-Day Tech Industry**: The tech industry, particularly in areas like Silicon Valley, has seen significant consolidation, with a few large firms like Google, Amazon, and Facebook dominating the market. This has led to concerns about monopsony power and its impact on workers.\n\nCommon prerequisites for monopsony power to emerge include:\n\n* **Market concentration**: A decrease in the number of firms operating in a market can lead to increased concentration and monopsony power.\n\n* **Barriers to entry**: High barriers to entry, such as significant startup costs or regulatory hurdles, can prevent new firms from entering the market and challenging existing firms.\n\n* **Lack of worker mobility**: When workers have limited ability to move between firms or industries, they are more vulnerable to monopsony power.\n\n* **Weak labor laws and regulations**: Inadequate labor laws and regulations can allow firms to exploit workers and maintain monopsony power.\n\n* **Globalization and technological change**: Shifts in global trade patterns and technological advancements can lead to industry consolidation and increased monopsony power.\n\nThese factors can create an environment where monopsony power can emerge and persist.', role='assistant', executed_tools=None, function_call=None, reasoning=None, tool_calls=None))], created=1751977661, model='meta-llama/llama-4-scout-17b-16e-instruct', object='chat.completion', system_fingerprint='fp_79da0e0073', usage=CompletionUsage(completion_tokens=387, prompt_tokens=2436, total_tokens=2823, completion_time=0.783724838, prompt_time=0.073756772, queue_time=0.211007971, total_time=0.85748161), usage_breakdown=None, x_groq={'id': 'req_01jzn034etfkva86zyj3y8x3kg'})

    # for response.choices[0].message
    # ChatCompletionMessage(content="A monopsony and a monopoly are similar in that both refer to a market structure where one entity has significant control. However, a monopoly refers to a single seller dominating a market, while a monopsony refers to a single buyer dominating a market.\n\nHere are some historical examples of monopsony-like structures consolidating over time:\n\n### Historical Examples \n\n1. **The company town phenomenon**: In the late 19th and early 20th centuries, many company towns in the United States were dominated by a single employer, such as a mining or manufacturing company. The company controlled not only the jobs but also the housing, stores, and other services, creating a monopsony-like situation.\n2. **The coal mining industry in Appalachia**: During the early 20th century, coal mining companies in Appalachia dominated the labor market, controlling wages, working conditions, and even the local economy. This created a monopsony-like situation, where workers had limited job opportunities and were subject to exploitation.\n3. **The farmworker situation in California**: In the mid-20th century, large agricultural companies in California, such as the growers' associations, dominated the labor market for farmworkers. This led to low wages, poor working conditions, and limited job opportunities for workers.\n\n### Common Prerequisites \n\nSome common circumstances that can lead to a monopsony-like structure include:\n\n* **Economic concentration**: When a few large companies dominate an industry, it can create a monopsony-like situation.\n* **Limited job opportunities**: When there are few job opportunities in a particular region or industry, workers may be more vulnerable to exploitation by a single employer.\n* **Lack of worker mobility**: When workers have limited ability to move to other regions or industries, they may be more susceptible to a monopsonistic employer.\n* **Weak labor unions**: When labor unions are weak or absent, workers may have limited bargaining power to negotiate better wages and working conditions.\n* **Government policies**: Government policies, such as lax antitrust enforcement or weak labor laws, can contribute to the creation of a monopsony-like structure.\n\n### Examples from Specific Industries \n\n* **The steel industry in the United States**: In the late 19th and early 20th centuries, a few large steel companies, such as U.S. Steel, dominated the industry and exerted significant control over workers.\n* **The technology industry in Silicon Valley**: Some argue that large tech companies, such as Google and Facebook, have created a monopsony-like situation in the tech industry, with limited job opportunities and high barriers to entry for new companies.\n\nThese examples illustrate how monopsony-like structures can emerge in different industries and contexts, often as a result of a combination of economic, social, and policy factors.", role='assistant', executed_tools=None, function_call=None, reasoning=None, tool_calls=None)

    for model_response in model_responses:
        print("model_response", model_response)

    # model_responses_path = Path.cwd() / "model_responses.json"
    # JSONFile.save_json(model_responses_path, model_responses)

@pytest.mark.skipif(
    not path_for_prompts_collection().exists(),
    reason="Prompts collection not found locally")
def test_on_alex_prompter_posts():

    parse_prompts_collection = ParsePromptsCollection(
        path_for_prompts_collection())
    lines_of_files = parse_prompts_collection.load_manually_copied_X_posts()
    posts = parse_prompts_collection.parse_manually_copied_X_posts(
        lines_of_files)

    chat_completion_configuration = ChatCompletionConfiguration()
    chat_completion_configuration.model = \
        "meta-llama/llama-4-scout-17b-16e-instruct"
    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.configuration = chat_completion_configuration

    conversation_and_system_messages = ConversationAndSystemMessages()
    assert conversation_and_system_messages.system_messages_manager.messages == []

    user_prompt_and_model_responses = []
    user_prompt_and_model_responses_path = \
        Path.cwd() / "user_prompt_and_model_responses.txt"

    try:
        for post in posts:
            conversation_and_system_messages.append_message(
                UserMessage(content=post["prompt"]))
            user_prompt_and_model_responses.append(
                f"User prompt: {post['prompt']}\n")

            conversation_as_input = \
                conversation_and_system_messages.get_conversation_as_list_of_dicts()
            response = groq_api_wrapper.create_chat_completion(
                conversation_as_input)

            if response and hasattr(response, "choices") and \
                len(response.choices) > 0:
                assistant_message_content = response.choices[0].message.content
                conversation_and_system_messages.append_message(
                    AssistantMessage(content=assistant_message_content))
                # Uncomment to print the assistant message content
                #print(assistant_message_content)
                
                user_prompt_and_model_responses.append(
                    f"Model response: {assistant_message_content}\n")
            else:
                print(f"No response message received from API call: {response}")
                conversation_and_system_messages.append_general_message(
                    response)
                print("response", response)
        TextFile.save_lines(
            user_prompt_and_model_responses_path,
            user_prompt_and_model_responses)
    except Exception as e:
        print(f"Error: {e}")
        if len(user_prompt_and_model_responses) > 0:
            TextFile.save_lines(
                user_prompt_and_model_responses_path,
                user_prompt_and_model_responses)
        else:
            print("user_prompt_and_model_responses is empty")
        raise
