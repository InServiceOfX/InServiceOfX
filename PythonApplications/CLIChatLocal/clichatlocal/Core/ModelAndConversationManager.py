from commonapi.FileIO import SystemMessagesFileIO
from commonapi.Messages import (
    AssistantMessage,
    ConversationSystemAndPermanent,
    UserMessage)
from corecode.Configuration import ModelList
from moretransformers.Applications import ModelAndTokenizer

class ModelAndConversationManager:
    def __init__(self, app):
        self._app = app

        self._csp = ConversationSystemAndPermanent()

        self._application_paths = app._application_paths

        self._system_messages_file_io = None
        self._load_system_messages()

        self._model_list = ModelList.from_yaml(
            self._application_paths.configuration_file_paths["model_list"])

        # Explicitly run load_configurations_and_model() to load the model and
        # tokenizer (i.e. self._mat)
        self._mat = None

    def _load_system_messages(self):
        self._application_paths.create_missing_system_messages_file()
        self._system_messages_file_io = SystemMessagesFileIO(
            self._application_paths.system_messages_file_path)

        if not self._system_messages_file_io.is_file_path_valid():
            raise RuntimeError(
                f"System messages file path {self._application_paths.system_messages_file_path} is not valid")

        self._system_messages_file_io.load_messages()

        self._system_messages_file_io.put_messages_into_system_messages_manager(
            self._csp.casm.system_messages_manager)

        self._system_messages_file_io.put_messages_into_conversation_history(
            self._csp.casm.conversation_history)

    def _get_first_model_path(self):
        first_model_name = next(iter(self._model_list.models))
        first_model_path = self._model_list.models[first_model_name]

        if not first_model_path.exists():
            raise FileNotFoundError(
                f"Model path {first_model_path} does not exist")

        print(f"First model: {first_model_name}")
        print(f"First model path: {first_model_path}")
        return first_model_name, first_model_path

    def _load_model_configurations(self):
        if self._app._process_configurations.configurations is not None:
            return (
                self._app._process_configurations.configurations[
                    "from_pretrained_model_configuration"],
                self._app._process_configurations.configurations[
                    "from_pretrained_tokenizer_configuration"],
                self._app._process_configurations.configurations[
                    "generation_configuration"])
        else:
            self._app._process_configurations.process_configurations()
            return (
                self._app._process_configurations.configurations[
                    "from_pretrained_model_configuration"],
                self._app._process_configurations.configurations[
                    "from_pretrained_tokenizer_configuration"],
                self._app._process_configurations.configurations[
                    "generation_configuration"])

    def _load_model_and_tokenizer(
            self,
            model_path,
            from_pretrained_model_configuration = None,
            from_pretrained_tokenizer_configuration = None,
            generation_configuration = None):

        self._mat = ModelAndTokenizer(
            model_path=model_path,
            from_pretrained_model_configuration=\
                from_pretrained_model_configuration,
            from_pretrained_tokenizer_configuration=\
                from_pretrained_tokenizer_configuration,
            generation_configuration=generation_configuration)

        self._mat._fptc.pretrained_model_name_or_path = model_path
        self._mat._fpmc.pretrained_model_name_or_path = model_path

        self._mat.load_tokenizer()
        self._mat.load_model()

    def load_configurations_and_model(self):
        from_pretrained_model_configuration, \
            from_pretrained_tokenizer_configuration, \
            generation_configuration = self._load_model_configurations()

        _, model_path = self._get_first_model_path()    

        self._load_model_and_tokenizer(
            model_path,
            from_pretrained_model_configuration,
            from_pretrained_tokenizer_configuration,
            generation_configuration)

    def respond_to_user_message(self, user_message: str):
        user_message = UserMessage(user_message)
        self._csp.append_message(user_message)

        response = self._mat.apply_chat_template_and_generate(
            self._csp.get_conversation_as_list_of_dicts(),
            with_attention_mask=True)

        assistant_message = AssistantMessage(response)
        self._csp.append_message(assistant_message)

        return response

    def clear_conversation_history(self, is_keep_active_system_messages=True):
        self._csp.clear_conversation_history(
            is_keep_active_system_messages=is_keep_active_system_messages)