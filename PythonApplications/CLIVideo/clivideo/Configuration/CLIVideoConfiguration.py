from pathlib import Path
from prompt_toolkit.styles import Style
import yaml

class CLIVideoConfiguration:
    def __init__(self, configuration_path: Path):
        with open(configuration_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Hot keys configuration
        self.hotkeys = config.get('hotkeys', {})
        self.mode_switch_key = self.hotkeys.get('mode_switch', 'c-m')  # Default: Ctrl+M
        
        # Styling configuration
        style_config = config.get('style', {})
        self.toolbar_style = style_config.get('toolbar', {})
        self.mode_text_color = self.toolbar_style.get('mode_text', '#ansiwhite')
        self.mode_background = self.toolbar_style.get('background', '#ansiblack')
        
        # Create prompt_toolkit Style
        self.prompt_style = Style.from_dict({
            'bottom-toolbar': f'bg:{self.mode_background}',
            'bottom-toolbar.mode': f'fg:{self.mode_text_color} bold',
        })
