from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

def test_ChatPromptTemplate_constructs_from_messages():
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("user", "What is the weather in {location}?"),
        ]
    )

    assert isinstance(chat_prompt_template.messages[0], SystemMessagePromptTemplate)
    assert chat_prompt_template.messages[0].input_variables == []
    assert chat_prompt_template.messages[0].prompt.template == \
        "You are a helpful assistant."

    assert isinstance(chat_prompt_template.messages[1], HumanMessagePromptTemplate)
    assert chat_prompt_template.messages[1].input_variables == ["location"]
    assert chat_prompt_template.messages[1].prompt.template == \
        "What is the weather in {location}?"

    assert chat_prompt_template.validate_template == False

def test_invoke_invokes():
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("user", "What is the weather in {location}?"),
        ]
    )

    prompt_value = chat_prompt_template.invoke({"location": "San Francisco"})

    assert prompt_value.messages[1].content == "What is the weather in San Francisco?"
