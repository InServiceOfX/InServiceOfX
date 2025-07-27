from commonapi.Messages import (
    ConversationSystemAndPermanent,
    ParsePromptsCollection,
    AssistantMessage,
    UserMessage)
from corecode.FileIO import JSONFile
from corecode.Utilities import DataSubdirectories, is_model_there
from moretransformers.Configurations import (
    CreateDefaultGenerationConfigurations,
    FromPretrainedModelConfiguration,
    FromPretrainedTokenizerConfiguration,
)
from moretransformers.Utilities import (
    get_tokens_per_second_statistics,
    summarize_tokens_per_second_statistics)

from pathlib import Path

from transformers import (
    Qwen3ForCausalLM,
    Qwen2Tokenizer)

import pytest, time, torch

data_subdirectories = DataSubdirectories()

relative_model_path = "Models/LLM/Menlo/Lucy-128k"

is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

model_is_not_downloaded_message = f"Model {relative_model_path} not downloaded"

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_use_configurations():
    from_pretrained_tokenizer_configuration = FromPretrainedTokenizerConfiguration(
        pretrained_model_name_or_path=model_path)
    tokenizer = Qwen2Tokenizer.from_pretrained(
        **from_pretrained_tokenizer_configuration.to_dict())

    from_pretrained_model_configuration = FromPretrainedModelConfiguration(
        pretrained_model_name_or_path=model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2")
    model = Qwen3ForCausalLM.from_pretrained(
        **from_pretrained_model_configuration.to_dict())
    assert model.config.max_position_embeddings == 131072
    assert hasattr(model.config, "rope_scaling")

    generation_configuration = \
        CreateDefaultGenerationConfigurations.for_Menlo_Lucy_128k()

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

    statistics = []

    conversation_to_save = []

    for i in range(len(user_message_texts)):
        text = user_message_texts[i]
        print(text)
        csap.append_message(UserMessage(text))
        tokenizer_outputs = tokenizer.apply_chat_template(
            conversation=csap.get_conversation_as_list_of_dicts(),
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
            return_dict=True).to(model.device)
        input_token_count = tokenizer_outputs["input_ids"].shape[1]

        start_time = time.time()

        output = model.generate(
            input_ids=tokenizer_outputs["input_ids"],
            attention_mask=tokenizer_outputs["attention_mask"],
            **generation_configuration.to_dict())
        end_time = time.time()

        output_token_count = output.shape[1]

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        assert isinstance(response, str)
        print(response)
        csap.append_message(AssistantMessage(response))

        stats = get_tokens_per_second_statistics(
            input_token_count,
            output_token_count,
            start_time,
            end_time)

        for key, value in stats.items():
            print(f"{key}: {value}")
        statistics.append(stats)

        conversation_to_save.extend(csap.get_conversation_as_list_of_dicts())

        csap.clear_conversation_history(is_keep_active_system_messages=False)

    # Samity checks:
    # print(len(csap.get_conversation_as_list_of_dicts()))
    # print(len(statistics))
    # print(len(user_message_texts))
    # assert len(csap.get_conversation_as_list_of_dicts()) == \
    #     2 * len(user_message_texts)
    # assert len(statistics) == len(user_message_texts)

    summary = summarize_tokens_per_second_statistics(statistics)
    statistics.append(summary)

    JSONFile.save_json(
        Path.cwd() / "test_use_configurations.json",
        conversation_to_save)

    print(
        f"Saved conversation to {Path.cwd().resolve() / 'test_use_configurations.json'}")

    JSONFile.save_json(
        Path.cwd() / "test_use_configurations_statistics.json",
        statistics)

    print(
        f"Saved statistics to {Path.cwd().resolve() / 'test_use_configurations_statistics.json'}")