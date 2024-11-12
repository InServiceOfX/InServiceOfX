from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.prompt_values import ChatPromptValue

from langchain_core.messages import SystemMessage, HumanMessage

# Recall that MessagesPlaceholder(BaseMessagePromptTemplate) is a prompt
# template that assumes variable is already list of messages.
#
# It's a placeholder which can be used to pass in a list of messages.

def test_direct_usage():
    prompt = MessagesPlaceholder(variable_name="history")
    assert prompt.input_variables == ["history"]
    format_result = prompt.format_messages(
        history=[
            ("system", "You are an AI assistant."),
            ("human", "Hello!")])
    
    assert isinstance(format_result, list)
    assert len(format_result) == 2
    assert isinstance(format_result[0], SystemMessage)
    assert isinstance(format_result[1], HumanMessage)

    prompt = MessagesPlaceholder(variable_name="messages")
    format_result = prompt.format_messages(
        messages=[
            ("system", "You are an AI assistant."),
            ("user", "Hello!")])

    assert isinstance(format_result, list)
    assert len(format_result) == 2
    assert isinstance(format_result[0], SystemMessage)
    assert isinstance(format_result[1], HumanMessage)

def test_build_prompt_with_chat_history():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    assert prompt.input_variables == ["history", "question"]

    invoke_result = prompt.invoke(
        {
            "history": [("human", "what's 5 + 2"), ("ai", "5 + 2 is 7")],
            "question": "now multiply that by 4"
        })
    assert isinstance(invoke_result, ChatPromptValue)
    assert invoke_result.messages[0].content == "You are a helpful assistant."
    assert invoke_result.messages[1].content == "what's 5 + 2"
    assert invoke_result.messages[2].content == "5 + 2 is 7"
    assert invoke_result.messages[3].content == "now multiply that by 4"