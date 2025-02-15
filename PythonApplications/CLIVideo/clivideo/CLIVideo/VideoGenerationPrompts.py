from prompt_toolkit import PromptSession
from typing import Tuple, Optional

class VideoGenerationPrompts:
    def __init__(self, session: PromptSession):
        self.session = session
        self.commands = {
            '.add_image': self.handle_add_image,
            '.help': self.show_help
        }

    async def handle_command(self, command: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Handle video generation commands
        
        Returns:
            Tuple[bool, Optional[str], Optional[str]]: 
            (should_continue, url, prompt_description)
        """
        command = command.strip().lower()
        if command in self.commands:
            return await self.commands[command]()
        return True, None, None

    async def handle_add_image(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """Handle adding an image with optional description"""
        print("\n=== Add Image ===")
        url = await self.session.prompt_async(
            "Enter image URL: "
        )
        
        if not url.strip():
            print("No URL provided, canceling...")
            return True, None, None
            
        description = await self.session.prompt_async(
            "Enter optional description (press Enter to skip): "
        )
        
        return True, url.strip(), description.strip() if description.strip() else None

    async def show_help(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """Show available commands"""
        print("\nAvailable commands:")
        print("  .add_image - Add an image with optional description")
        print("  .help     - Show this help message")
        return True, None, None
