from corecode.FileIO import TextFile
from corecode.Utilities import (
    DataSubdirectories,
    git_clone_repo,
    _parse_repo_url_into_target_path,
)
from string import Template
from datetime import datetime
from typing import Optional

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

    class MetaAIWhatsAppTemplate:
        """Template for Meta AI WhatsApp system prompt with customizable
        parameters."""
        
        def __init__(self, template_string: str):
            self._template = Template(template_string)
        
        def format_prompt(
            self,
            model_name: str = "Meta AI",
            model_engine: str = "Llama 4", 
            company: str = "Meta",
            date: Optional[str] = None,
            location: str = "California"
        ) -> str:
            """
            Format the Meta AI WhatsApp prompt with customizable parameters.
            
            Args:
                model_name: Name of the AI model (default: "Meta AI")
                model_engine: Engine powering the model (default: "Llama 4")
                company: Company that made the AI (default: "Meta")
                date: Date string (default: current date from datetime)
                location: User location (default: "California")
                
            Returns:
                str: Formatted system prompt
            """
            # Use current date if none provided
            if date is None:
                date = datetime.now().strftime("%A, %B %d, %Y")
            
            # Create substitution dictionary
            substitutions = {
                'model_name': model_name,
                'model_engine': model_engine,
                'company': company,
                'date': date,
                'location': location
            }
            
            return self._template.substitute(substitutions)

    def _get_meta_ai_whatsapp_prompt(self):
        prompt_path = self._repo_path / "meta-ai-whatsapp_20250528.md"

        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found at {prompt_path}")

        text = TextFile.load_text(prompt_path)

        if text is None:
            raise FileNotFoundError(
                f"Prompt file is empty at {prompt_path}")

        start_marker = "You are an expert"
        start_index = text.find(start_marker)

        if start_index == -1:
            # If we can't find the expected start, return the original text
            return text

        # Return the system prompt.
        return text[start_index:]

    def get_meta_ai_whatsapp_string_template(self):
        system_prompt = self._get_meta_ai_whatsapp_prompt()
        
        # Convert the string into a template by replacing specific occurrences
        template_string = self._convert_meta_ai_whatsapp_to_template(system_prompt)
        
        return self.MetaAIWhatsAppTemplate(template_string)
    
    def _convert_meta_ai_whatsapp_to_template(self, prompt_text: str) -> str:
        """
        Convert the prompt text into a template string by replacing specific values.
        
        Args:
            prompt_text: Original prompt text
            
        Returns:
            str: Template string with placeholders
        """
        # Replace specific occurrences with template variables
        template_text = prompt_text
        
        template_text = template_text.replace(
            "made by Meta",
            "made by ${company}")
        
        template_text = template_text.replace(
            "Your name is Meta AI,",
            "Your name is ${model_name},")
        template_text = template_text.replace(
            "You are Meta AI and",
            "You are ${model_name} and")
        
        template_text = template_text.replace(
            "powered by Llama 4,",
            "powered by ${model_engine},")
        
        template_text = template_text.replace(
            "Wednesday, May 28, 2025",
            "${date}")
        
        template_text = template_text.replace(
            "The user is in Morocco.",
            "The user is in ${location}.")
        
        return template_text


