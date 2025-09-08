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

import pytest, time, torch

data_subdirectories = DataSubdirectories()

relative_model_path = "Models/LLM/Qwen/Qwen3-0.6B"

is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

model_is_not_downloaded_message = f"Model {relative_model_path} not downloaded"

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_use_configurations():
    from_pretrained_tokenizer_configuration = FromPretrainedTokenizerConfiguration(
        pretrained_model_name_or_path=model_path)

    from_pretrained_model_configuration = FromPretrainedModelConfiguration(
        pretrained_model_name_or_path=model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2")

    generation_configuration = \
        CreateDefaultGenerationConfigurations.for_Qwen3_thinking()

    model_and_tokenizer = ModelAndTokenizer(
        model_path=model_path,
        from_pretrained_model_configuration=from_pretrained_model_configuration,
        from_pretrained_tokenizer_configuration=from_pretrained_tokenizer_configuration,
        generation_configuration=generation_configuration)

    model_and_tokenizer.load_model()
    model_and_tokenizer.load_tokenizer()

    assert model_and_tokenizer._model.config.max_position_embeddings == \
        131072
    assert hasattr(model_and_tokenizer._model.config, "rope_scaling")

    user_message_texts = []

    # Setup conversation data.

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