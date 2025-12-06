from dataclasses import dataclass, asdict

@dataclass
class FunctionCallOutputMessage:
    """This is used when you use Response, instead of ChatCompletion, objects.
    OpenAI API:
    https://platform.openai.com/docs/guides/function-calling#handling-function-calls

    input_messages.append({
        "type": "function_call_output",
        "call_id": tool_call.call_id,
        "output": str(result)
    })

    https://platform.openai.com/docs/guides/function-calling#function-tool-example
    # 4. Provide function call results to the model
    input_list.append({
        "type": "function_call_output",
        "call_id": item.call_id,
        "output": json.dumps({
            "horoscope": horoscope
        })
    })
    """

    type: str = "function_call_output"
    call_id: str = None
    # This is typically the output from json.dumps
    output: str = None

    def to_dict(self) -> dict:
        return asdict(self)