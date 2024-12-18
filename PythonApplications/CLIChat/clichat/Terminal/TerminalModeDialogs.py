from prompt_toolkit import print_formatted_text, HTML
from prompt_toolkit.styles import Style
from prompt_toolkit.shortcuts import radiolist_dialog, checkboxlist_dialog


class TerminalModeDialogs:

    def __init__(self, parent, configuration) -> None:
        self.parent = parent
        self.style = Style.from_dict({
            "dialog": "bg:ansiblack",
            "dialog text-area": \
                f"bg:ansiblack {configuration.terminal_CommandEntryColor2}",
            "dialog text-area.prompt": \
                configuration.terminal_PromptIndicatorColor2,
            "dialog radio-checked": configuration.terminal_ResourceLinkColor,
            "dialog checkbox-checked": configuration.terminal_ResourceLinkColor,
            "dialog button.arrow": configuration.terminal_ResourceLinkColor,
            "dialog button.focused": \
                f"bg:{configuration.terminal_ResourceLinkColor} ansiblack",
            "dialog frame.border": configuration.terminal_ResourceLinkColor,
            "dialog frame.label": \
                f"bg:ansiblack {configuration.terminal_ResourceLinkColor}",
            "dialog.body": "bg:ansiblack ansiwhite",
            "dialog shadow": "bg:ansiblack",}
        ) if configuration.terminalResourceLinkColor.startswith("ansibright") \
            else Style.from_dict({
                "dialog": "bg:ansiwhite",
                "dialog text-area": \
                    f"bg:ansiblack {configuration.terminal_CommandEntryColor2}",
                "dialog text-area.prompt": 
                    configuration.terminal_PromptIndicatorColor2,
                "dialog radio-checked": configuration.terminal_ResourceLinkColor,
                "dialog checkbox-checked": \
                    configuration.terminal_ResourceLinkColor,
                "dialog button.arrow": configuration.terminal_ResourceLinkColor,
                "dialog button.focused": \
                    f"bg:{configuration.terminal_ResourceLinkColor} ansiblack",
                "dialog frame.border": configuration.terminal_ResourceLinkColor,
                "dialog frame.label": \
                    f"bg:ansiwhite {configuration.terminal_ResourceLinkColor}",
                "dialog.body": "bg:ansiwhite ansiblack",
                "dialog shadow": "bg:ansiwhite",
            }
        )

        self._configuration = configuration

    def get_valid_options(
            self, 
            options: list[str] = [], 
            descriptions: list[str] = [], 
            bold_descriptions: bool = False, 
            filter: str = "", 
            default: str = "", 
            title: str = "Available Options", 
            text: str = "Select an option:"
        ) -> str:
        if not options:
            return ""
        filter = filter.strip().lower()
        if descriptions:
            lowercase_descriptions = [i.lower() for i in descriptions]
            values = [
                (
                    option,
                    HTML(f"<b>{descriptions[index]}</b>") \
                        if bold_descriptions else descriptions[index]) \
                            for index, option in enumerate(options) \
                                if (filter in option.lower() or \
                                    filter in lowercase_descriptions[index])]
        else:
            values = [
                (option, option) for option in options \
                    if filter in option.lower()]
        if not values:
            if descriptions:
                values = [
                    (
                        option,
                        HTML(f"<b>{descriptions[index]}</b>") \
                            if bold_descriptions else descriptions[index]) \
                                for index, option in enumerate(options)]
            else:
                values = [(option, option) for option in options]
        result = radiolist_dialog(
            title=title,
            text=text,
            values=values,
            default=default if default and default in options else values[0][0],
            style=self.style,
        ).run()
        if result:
            notice = f"You've chosen: {result}"
            splitted_content = notice.split(": ", 1)
            key, value = splitted_content
            print_formatted_text(
                HTML(
                    f"<{self._configuration.terminal_PromptIndicatorColor2}>{key}:</{self._configuration.terminal_PromptIndicatorColor2}> {value}"))
            return result
        return ""

    def display_feature_menu(self, heading: str, features: list[str]):
        values = [
            (
                command,
                command if self._configuration.terminal_DisplayCommandOnMenu \
                    else self.parent.dotCommands[command][0]) \
                        for command in features]
        result = radiolist_dialog(
            title=heading,
            text="Select a feature:",
            values=values,
            default=features[0],
            style=self.style,
        ).run()
        if result:
            self.parent.printRunningCommand(result)
            return self.parent.getContent(result)
        else:
            print_formatted_text(
                HTML(
                    f"<{self._configuration.terminal_PromptIndicatorColor2}>"
                    "Action cancelled"
                    f"</{self._configuration.terminal_PromptIndicatorColor2}>"))
            return ""

    def get_multiple_selection(
            self, 
            title: str = "Multiple Selection", 
            text: str = "Select item(s):", 
            options: list[str] = ["ALL"], 
            descriptions: list[str] = [], 
            default_values: list[str] = ["ALL"]):
        if descriptions:
            values = [
                (option, descriptions[index]) \
                    for index, option in enumerate(options)]
        else:
            values = [(option, option) for option in options]
        return checkboxlist_dialog(
            title=title,
            text=text,
            values=values,
            default_values=default_values,
            style=self.style,
        ).run()