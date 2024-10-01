from moretransformers.Conversation.StatementTypes import (
    SystemStatement,
    UserStatement,
    AssistantStatement)

def test_SystemStatement_instantiates():
    statement = SystemStatement()
    assert isinstance(statement, dict)

def test_UserStatement_instantiates():
    statement = UserStatement()
    assert isinstance(statement, dict)

def test_AssistantStatement_instantiates():
    statement = AssistantStatement()
    assert isinstance(statement, dict)

def test_add_statements_as_dicts_to_conversation():
    conversation = []

    system_prompt = "You are a helpful assistant"
    user_prompt = "What is the capital of the moon?"

    conversation.append(SystemStatement(system_prompt).to_dict())

    conversation.append(UserStatement(user_prompt).to_dict())

    assert len(conversation) == 2
    assert isinstance(conversation[0], dict)
    assert isinstance(conversation[1], dict)

    assert "role" in conversation[0].keys()
    assert "content" in conversation[0].keys()
    assert "role" in conversation[1].keys()
    assert "content" in conversation[1].keys()

    assert conversation[0]["role"] == "system"
    assert conversation[0]["content"] == system_prompt
    assert conversation[1]["role"] == "user"
    assert conversation[1]["content"] == user_prompt

    assistant_response = "The capital of the moon is called New Moon."

    conversation.append(AssistantStatement(assistant_response).to_dict())

    assert len(conversation) == 3
    assert isinstance(conversation[2], dict)

    assert "role" in conversation[2].keys()
    assert "content" in conversation[2].keys()

    assert conversation[2]["role"] == "assistant"
    assert conversation[2]["content"] == assistant_response





