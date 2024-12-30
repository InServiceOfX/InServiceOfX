from clichat.Configuration import CLIChatConfiguration

def test_CLIChatConfiguration_inits_with_defaults():
    config = CLIChatConfiguration()
    
    # Test default values
    assert config.temperature == 1.0
    assert config.terminal_CommandEntryColor2 == "ansigreen"
    assert config.terminal_PromptIndicatorColor2 == "ansicyan"
    assert config.terminal_ResourceLinkColor == "ansiyellow"
