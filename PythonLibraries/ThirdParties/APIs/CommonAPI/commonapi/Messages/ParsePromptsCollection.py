from corecode.FileIO import TextFile
from pathlib import Path
from warnings import warn
from typing import List, Dict, Optional
import re

class ParsePromptsCollection:
    USER_X_RELATIVE_PATHS = [
        "User/X/alex_prompter_2025-07-10_I_tested_G.txt",
    ]

    def __init__(self, prompts_collection_path: str | Path):
        self._prompts_collection_path = Path(prompts_collection_path)

    def load_manually_copied_X_posts(
            self,
            list_of_str_paths: list[str] | None = None):

        if list_of_str_paths is None:
            list_of_str_paths = self.USER_X_RELATIVE_PATHS

        lines_of_files = []
        for relative_path in list_of_str_paths:
            file_path = self._prompts_collection_path / relative_path
            if file_path.exists():
                lines = TextFile.load_lines(file_path)
                if lines is not None:
                    lines_of_files.append(lines)
                else:
                    warn(f"File {file_path} is empty")
            else:
                warn(f"File {file_path} does not exist")
        return lines_of_files

    def parse_manually_copied_X_posts(
            self,
            lines_of_files: List[List[str]]) -> List[Dict[str, str]]:
        """
        Args:
            lines_of_files: List of file lines from
            load_manually_copied_X_posts()
        """
        all_posts = []
        
        for file_lines in lines_of_files:
            posts = self._parse_single_file_of_X_posts(file_lines)
            all_posts.extend(posts)
            
        return all_posts

    def _parse_single_file_of_X_posts(
            self,
            lines: List[str]) -> List[Dict[str, str]]:
        """        
        Args:
            lines: Lines from a single file of X posts
        """
        posts = []
        current_post = {"url": "", "section_title": "", "prompt": ""}
        in_prompt_section = False
        prompt_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue

            # Check if this line starts a new post
            if self._is_new_post_start(line):

                # Save previous post if it has content
                if current_post["section_title"] \
                    and current_post["section_title"] != '' \
                    and prompt_lines is not None and prompt_lines != []:
                    current_post["prompt"] = '\n'.join(prompt_lines).strip()
                    posts.append(current_post)
                
                    # Start new post
                    current_post = \
                        {"url": "", "section_title": "", "prompt": ""}
                    in_prompt_section = False
                    prompt_lines = []
                
                # Parse URL if present
                if line.startswith('https://x.com'):
                    current_post["url"] = line
                elif self._is_section_title(line):
                    current_post["section_title"] = line
                
            # Check if this is the "Prompt:" marker
            elif line == "Prompt:":
                in_prompt_section = True
                
            # Check if we're in prompt section and hit the arrow (end of prompt)
            elif in_prompt_section and line.startswith('→'):
                # End of prompt section
                in_prompt_section = False
                
            # Collect prompt lines
            elif in_prompt_section:
                prompt_lines.append(line)

        # Don't forget the last post
        if current_post["section_title"] or current_post["prompt"]:
            current_post["prompt"] = '\n'.join(prompt_lines).strip()
            posts.append(current_post)
            
        return posts

    def _is_new_post_start(self, line: str) -> bool:
        """
        Check if a line indicates the start of a new post.
        """
        # Check for URL pattern
        if line.startswith('https://x.com'):
            return True
        
        # Check for numbered section title pattern
        if self._is_section_title(line):
            return True
            
        return False

    def _is_section_title(self, line: str) -> bool:
        """
        Check if a line is a section title.
        """
        # Pattern: number followed by dot and title
        pattern = r'^\d+\.\s+.+'
        return bool(re.match(pattern, line))


