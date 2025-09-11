from commonapi.Messages import (
    ConversationSystemAndPermanent,
    ParsePromptsCollection,
    AssistantMessage,
    UserMessage)

from corecode.FileIO import JSONFile
from corecode.Statistics import (
    get_tokens_per_second_statistics,
    PerformanceMetrics,
    summarize_tokens_per_second_statistics,
    )
from corecode.Utilities import DataSubdirectories, is_model_there
from moretransformers.Applications import ModelAndTokenizer
from moretransformers.Configurations import (
    CreateDefaultGenerationConfigurations,
    FromPretrainedModelConfiguration,
    FromPretrainedTokenizerConfiguration,)
from pathlib import Path

import pytest, torch

data_subdirectories = DataSubdirectories()

relative_model_path = "Models/LLM/Qwen/Qwen3-0.6B"

is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

model_is_not_downloaded_message = f"Model {relative_model_path} not downloaded"

def create_configurations_for_test():
    from_pretrained_tokenizer_configuration = \
        FromPretrainedTokenizerConfiguration(
            pretrained_model_name_or_path=model_path)
    from_pretrained_model_configuration = FromPretrainedModelConfiguration(
        pretrained_model_name_or_path=model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2")
    generation_configuration = \
        CreateDefaultGenerationConfigurations.for_Qwen3_thinking()
    return (
        from_pretrained_tokenizer_configuration,
        from_pretrained_model_configuration,
        generation_configuration)

def setup_conversation_data():
    user_message_texts = []
    if data_subdirectories.PromptsCollection.exists():
        parse_prompts_collection = ParsePromptsCollection(
            data_subdirectories.PromptsCollection)
        lines_of_files = parse_prompts_collection.load_manually_copied_X_posts()
        posts = parse_prompts_collection.parse_manually_copied_X_posts(
            lines_of_files)
        for post in posts:
            user_message_texts.append(post["prompt"])
    else:
        print(
            f"Prompts collection path not found in {data_subdirectories.PromptsCollection}")
        user_message_texts.append("What is C. elegans?")

    csap = ConversationSystemAndPermanent()
    return csap, user_message_texts

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_enable_thinking_true():
    from_pretrained_tokenizer_configuration, from_pretrained_model_configuration, generation_configuration = \
        create_configurations_for_test()

    mat = ModelAndTokenizer(
        model_path=model_path,
        from_pretrained_model_configuration=from_pretrained_model_configuration,
        from_pretrained_tokenizer_configuration=from_pretrained_tokenizer_configuration,
        generation_configuration=generation_configuration)

    mat.load_model()
    mat.load_tokenizer()

    assert mat._model.config.max_position_embeddings == 40960
    assert hasattr(mat._model.config, "rope_scaling")

    # Setup conversation data.
    csap, user_message_texts = setup_conversation_data()

    statistics = []
    conversation_to_save = []
    # Collect all metrics for summary
    all_metrics = []

    for i in range(len(user_message_texts)):
        text = user_message_texts[i]
        print(text)
        csap.append_message(UserMessage(text))
        tokenizer_outputs = mat.apply_chat_template(
            conversation=csap.get_conversation_as_list_of_dicts(),
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
            return_dict=True)
        input_token_count = tokenizer_outputs["input_ids"].shape[1]

        # Initialize PerformanceMetrics for this generation
        performance_metrics = PerformanceMetrics()
        performance_metrics.start_timing()

        generated_ids = mat.generate(
            tokenizer_outputs["input_ids"],
            attention_mask=tokenizer_outputs["attention_mask"])

        performance_metrics.end_timing()
        performance_metrics.record_memory_usage()

        output_token_count = generated_ids.shape[1]

        thinking_content, content = mat._parse_generate_output_into_thinking_and_content(
            tokenizer_outputs,
            generated_ids)

        print(f"Thinking content: {thinking_content}")
        print(f"Content: {content}")

        csap.append_message(AssistantMessage(content))

        # Get comprehensive metrics
        metrics = performance_metrics.get_metrics(
            input_token_count,
            output_token_count)
        
        # Collect metrics for summary
        all_metrics.append(metrics)
        
        # Add to statistics (keeping your existing format)
        statistics.append(get_tokens_per_second_statistics(
            input_token_count,
            output_token_count,
            performance_metrics.start_time,
            performance_metrics.end_time))

        conversation_to_save.extend(csap.get_conversation_as_list_of_dicts())
        csap.clear_conversation_history(is_keep_active_system_messages=False)

    # Print summary of all performance metrics
    print("\n" + "="*60)
    print("PERFORMANCE METRICS SUMMARY")
    print("="*60)
    
    summary = PerformanceMetrics.summarize_metrics(all_metrics)
    PerformanceMetrics.print_summary(summary)

    summary = summarize_tokens_per_second_statistics(statistics)
    statistics.append(summary)

    JSONFile.save_json(
        Path.cwd() / "test_enable_thinking_true.json",
        conversation_to_save)

    print(
        f"Saved conversation to {Path.cwd().resolve() / 'test_enable_thinking_true.json'}")

    JSONFile.save_json(
        Path.cwd() / "test_enable_thinking_true_statistics.json",
        statistics)

    print(
        f"Saved statistics to {Path.cwd().resolve() / 'test_enable_thinking_true_statistics.json'}")