import textwrap
from clichat.Utilities import Printing
from clichat.Utilities import ConfigureKeyBindings

from prompt_toolkit import prompt
from prompt_toolkit.filters import Condition


class SinglePrompt:

    @staticmethod
    def run(
        configuration,
        input_indicator="", 
        validator=None, 
        default="", 
        accept_default=False, 
        completer=None, 
        prompt_session=None, 
        style=None, 
        is_password=False, 
        bottom_toolbar=None, 
        **kwargs):
        """
        From
        https://python-prompt-toolkit.readthedocs.io/en/stable/pages/reference.html#prompt_toolkit.shortcuts.prompt
        recall
        prompt_toolkit.shortcuts.prompt(
            message: AnyFormattedText | None = None, *, 
            history: History | None = None,
            editing_mode: EditingMode | None = None, refresh_interval: float | None = None, vi_mode: bool | None = None, lexer: Lexer | None = None, completer: Completer | None = None, complete_in_thread: bool | None = None, is_password: bool | None = None, key_bindings: KeyBindingsBase | None = None, bottom_toolbar: AnyFormattedText | None = None, style: BaseStyle | None = None, color_depth: ColorDepth | None = None, cursor: AnyCursorShapeConfig = None, include_default_pygments_style: FilterOrBool | None = None, style_transformation: StyleTransformation | None = None, swap_light_and_dark_colors: FilterOrBool | None = None, rprompt: AnyFormattedText | None = None, multiline: FilterOrBool | None = None, prompt_continuation: PromptContinuationText | None = None, wrap_lines: FilterOrBool | None = None, enable_history_search: FilterOrBool | None = None, search_ignore_case: FilterOrBool | None = None, complete_while_typing: FilterOrBool | None = None, validate_while_typing: FilterOrBool | None = None, complete_style: CompleteStyle | None = None, auto_suggest: AutoSuggest | None = None, validator: Validator | None = None, clipboard: Clipboard | None = None, mouse_support: FilterOrBool | None = None, input_processors: list[Processor] | None = None, placeholder: AnyFormattedText | None = None, reserve_space_for_menu: int | None = None, enable_system_prompt: FilterOrBool | None = None, enable_suspend: FilterOrBool | None = None, enable_open_in_editor: FilterOrBool | None = None, tempfile_suffix: str | Callable[[], str] | None = None, tempfile: str | Callable[[], str] | None = None, default: str = '', accept_default: bool = False, pre_run: Callable[[], None] | None = None, set_exception_handler: bool = True, handle_sigint: bool = True, in_thread: bool = False, inputhook: InputHook | None = None) → str

        where
        message - Plain text or formatted text to be shown before the prompt.
        This can also be a callable that returns formatted text.
        validator - Validator instance for input validation.
        completer - Completer instance for input completion.
        style - Style instance for color scheme.
        swap_light_and_dark_colors - bool or Filter. When enabled, apply
        SwapLightAndDarkStyleTransformation; useful for switching between dark
        and light terminal backgrounds.
        bottom_toolbar - Formatted text or callable which is supposed to return 
        formatted text.
        """

        key_binder = ConfigureKeyBindings(configuration)
        this_key_bindings = key_binder.configure_key_bindings()

        config.selectAll = False
        input_prompt = prompt_session.prompt \
            if prompt_session is not None else prompt
        if not input_indicator:
            input_indicator = [("class:indicator", ">>> "),]
        user_input = input_prompt(
            input_indicator,
            key_bindings=this_key_bindings,
            bottom_toolbar=bottom_toolbar if bottom_toolbar is not None \
                else f""" {str(configuration.hotkey_exit).replace("'", "")} {configuration.exit_entry}""",
            swap_light_and_dark_colors=Condition(
                lambda: not configuration.terminal_ResourceLinkColor.startswith(
                    "ansibright")),
            style=style,
            validator=validator,
            multiline=Condition(lambda: hasattr(config, "currentMessages") and config.multilineInput),
            default=default,
            accept_default=accept_default,
            completer=completer,
            is_password=is_password,
            mouse_support=False,
            **kwargs,)
        
        # dedent to work with code block
        user_input = textwrap.dedent(user_input)
        return user_input \
            if hasattr(configuration, "addPathAt") and configuration.addPathAt \
                else user_input.strip()
