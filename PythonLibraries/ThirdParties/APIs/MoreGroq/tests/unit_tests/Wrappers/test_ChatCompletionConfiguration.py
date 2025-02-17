from moregroq.Wrappers.ChatCompletionConfiguration import (
    ChatCompletionConfiguration,
    FunctionDefinition,
    Tool
)

def test_Tool_becomes_dict():
    tool = Tool(
        type="function",
        function=FunctionDefinition(
            name="calculate",
            description="Evaluate a mathematical expression",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],            
            },
        ),
    )
    tools = [tool,]
    configuration = ChatCompletionConfiguration(
        tools=tools,
        tool_choice="auto"
    )

    assert configuration.to_dict() == {
        "model": "llama-3.3-70b-versatile",
        "n": 1,
        "stream": False,
        "temperature": 1.0,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Evaluate a mathematical expression",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression to evaluate",
                            }
                        },
                        "required": ["expression"],
                    }
                }
            }
        ],
        "tool_choice": "auto",
    }