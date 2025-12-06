from moretransformers.Tools import ToolCallProcessor

from textwrap import dedent

def test_ToolCallProcessor_has_nonempty_tool_call_works():
    example_1 = (
        "For each function call, return a json object with function name and "
        "arguments within <tool_call></tool_call> XML tags:")

    example_2 = dedent("""<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
""")

    example_3 = dedent("""For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
user
Hey, what's the temperature in Paris right now?
assistant""")

    assert not ToolCallProcessor.has_nonempty_tool_call(example_1)
    assert ToolCallProcessor.has_nonempty_tool_call(example_2)
    assert ToolCallProcessor.has_nonempty_tool_call(example_3)