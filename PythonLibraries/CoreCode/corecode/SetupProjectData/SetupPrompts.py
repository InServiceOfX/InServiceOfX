from corecode.FileIO import TextFile
from corecode.Utilities import (
    DataSubdirectories,
    git_clone_repo,
    _parse_repo_url_into_target_path,
)

def get_prompts_path():
    data_subdirectories = DataSubdirectories()

    data_path = data_subdirectories.get_data_path(0)
    prompts_path = data_path / "Prompts"

    if not prompts_path.exists():
        for path in data_subdirectories.DataPaths:
            prompts_path = path / "Prompts"
            if prompts_path.exists():
                break
        if not prompts_path.exists():
            raise FileNotFoundError(
                f"Prompts path not found in {data_subdirectories.DataPaths}")

    return prompts_path

def git_clone_repo_into_prompts_path(repo_url: str):
    prompts_path = get_prompts_path()
    git_clone_repo(repo_url, prompts_path)

# https://github.com/jujumilk3/leaked-system-prompts.git
class ParseJujumilk3LeakedSystemPrompts:
    _GIT_REPO_URL = "https://github.com/jujumilk3/leaked-system-prompts.git"

    def __init__(self):
        self._prompts_path = get_prompts_path()
        self._repo_path, _ = _parse_repo_url_into_target_path(
            self._GIT_REPO_URL,
            self._prompts_path,
        )

    def get_anthropic_claude_3_7_sonnet_prompt(self):
        prompt_path = self._repo_path / "anthropic-claude-3.7-sonnet_20250224.md"

        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found at {prompt_path}")

        text = TextFile.load_text(prompt_path)

        if text is None:
            raise FileNotFoundError(
                f"Prompt file is empty at {prompt_path}")

        start_marker = "The assistant is Claude,"
        start_index = text.find(start_marker)

        if start_index == -1:
            # If we can't find the expected start, return the original text
            return text

        system_prompt = text[start_index:]

        # Split by lines to handle the structure
        lines = text.split('\n')    
        file_stem_name = None
        html_address = None
        for line in lines:
            if line.startswith("# "):
                file_stem_name = line[2:].strip()
            if line.startswith("source: "):
                html_address = line[7:].strip()

        return system_prompt, file_stem_name, html_address


