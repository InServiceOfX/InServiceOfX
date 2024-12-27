from prompt_toolkit.shortcuts import checkboxlist_dialog, radiolist_dialog
from prompt_toolkit.styles import Style
from typing import List, Tuple, Optional
from clichat.Persistence import SystemMessage, SystemMessagesManager

def show_system_message_dialog(
    system_message_manager: SystemMessagesManager,
    dialog_style: Style) -> Optional[str]:
    
    values = [
        (msg.hash, msg.content)
        for msg in system_message_manager.messages
    ]
    
    default_values = [
        msg.hash 
        for msg in system_message_manager.get_active_messages()
    ]

    selected_hashes = checkboxlist_dialog(
        title="System Messages",
        text="Select active system messages:",
        values=values,
        default_values=default_values,
        style=dialog_style
    ).run()

    if selected_hashes is not None:
        # Update active states
        changes_made = False
        for msg in system_message_manager.messages:
            if msg.is_active != (msg.hash in selected_hashes):
                system_message_manager.toggle_message(msg.hash)
                changes_made = True
        
        if changes_made:
            # Show options for conversation management
            return radiolist_dialog(
                title="System Messages Updated",
                text="What would you like to do with the conversation?",
                values=[
                    (
                        "reset",
                        "Reset conversation (keep only active system messages)"
                    ),
                    (
                        "append",
                        "Append active system messages to current conversation"
                    ),
                    ("nothing", "Do nothing")],
                style=dialog_style
            ).run()
    
    return None