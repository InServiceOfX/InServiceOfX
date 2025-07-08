from corecode.SetupProjectData.SetupPrompts import \
    ParseJujumilk3LeakedSystemPrompts

import pytest

def path_for_example_dataset_0():
    return ParseJujumilk3LeakedSystemPrompts()._repo_path

@pytest.mark.skipif(
    not path_for_example_dataset_0().exists(),
    reason="Repository jujumilk3/leaked-system-prompts not found locally"
)
def test_ParseJujumilk3LeakedSystemPrompts():
    parse_jujumilk3_leaked_system_prompts = ParseJujumilk3LeakedSystemPrompts()
    system_prompt, file_stem_name, html_address = \
        parse_jujumilk3_leaked_system_prompts.get_anthropic_claude_3_7_sonnet_prompt()
    assert system_prompt is not None
    assert file_stem_name == "anthropic-claude-3.7-sonnet_20250224"
    assert html_address == \
        "<https://x.com/elder_plinius/status/1894173986151358717>"

    print(f"system_prompt: {system_prompt}")