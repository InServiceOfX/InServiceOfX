class SystemMessagesDialogHandler:
    def __init__(self, messages_manager: SystemMessagesManager, configuration):
        self.messages_manager = messages_manager
        self.configuration = configuration
        
    def show_active_system_messages(self, configuration):
        active_messages = self.get_active_messages()
        if not active_messages:
            print_formatted_text(
                HTML(
                    f"<{configuration.terminal_PromptIndicatorColor2}>"
                    "No active system messages"
                    f"</{configuration.terminal_PromptIndicatorColor2}>"))
            return

        print_formatted_text(
            HTML(
                f"<{configuration.terminal_SystemMessageColor}>"
                "Active system messages:\n"
                f"</{configuration.terminal_SystemMessageColor}>"))
        
        for msg in active_messages:
            print_formatted_text(
                HTML(
                    f"<{configuration.terminal_SystemMessageColor}>"
                    f"{configuration.terminal_SystemMessagePrefix} {msg.content}"
                    f"</{configuration.terminal_SystemMessageColor}>"))

    def add_system_message_dialog(self, dialog_style: Style) -> bool:
        
        # Assume multiline is false.
        message_content = prompt(
            "Enter new system message:\n",
            style=dialog_style)
        
        if not message_content.strip():
            return False
            
        # Show preview and confirm
        print("\nPreview of system message:")
        print("-" * 40)
        print(message_content)
        print("-" * 40)

        # https://python-prompt-toolkit.readthedocs.io/en/stable/pages/reference.html#prompt_toolkit.shortcuts.confirm
        # confirm(message: str = 'Confirm?', suffix: str = ' (y/n) ') -> bool
        if confirm(
            "Add this system message and make it active?"):
            new_message = self.add_message(message_content, True)
            return new_message is not None
        
        return False
        
    def handle_exit(self, configuration):
        
        if not self.messages:  # Don't offer to save if no messages
            return
        
        if configuration is None:
            # Ask to save in current directory
            if yes_no_dialog(
                title="Save System Messages",
                text="Would you like to save system messages in the current directory?"
            ).run():
                self.save_messages(Path.cwd() / "system_messages.json")
            return

        path = None
        try:
            path = get_path_from_configuration(configuration, "system_messages_path")
        except FileNotFoundError:
            if yes_no_dialog(
                title="Save System Messages",
                text=f"System messages path {configuration.system_messages_path} doesn't exist. Save in current directory?"
            ).run():
                self.save_messages(Path.cwd() / "system_messages.json")
                return
            
        # Path exists - ask to save/merge
        if yes_no_dialog(
            title="Save System Messages",
            text="Would you like to save/merge system messages?"
        ).run():
            try:
                # Try to load existing messages
                existing_manager = SystemMessagesManager()
                existing_manager.load_messages(path)
                    
                # Add only new messages
                for msg in self.messages:
                    if msg.hash not in existing_manager._messages_dict:
                        existing_manager._messages_dict[msg.hash] = msg
                    
                # Save merged messages
                existing_manager.save_messages(path)
                    
            except json.JSONDecodeError:
                # If JSON loading fails, because the file contents are not in
                # valid JSON format, just save current messages over the file.
                self.save_messages(path)
