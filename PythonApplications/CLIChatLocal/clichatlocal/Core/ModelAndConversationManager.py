from clichatlocal.Configuration import ModelList
from commonapi.Messages import (
    AssistantMessage,
    ConversationSystemAndPermanent,
    UserMessage)
from moretransformers.Applications import ModelAndTokenizer
from moretransformers.Configurations import (
    FromPretrainedModelConfiguration,
    FromPretrainedTokenizerConfiguration,
    GenerationConfiguration)

class ModelAndConversationManager:
    def __init__(self, app):
        self._application_paths = app._application_paths

        self._model_list = ModelList.from_yaml(
            self._application_paths.configuration_file_paths["model_list"])

        self._mat = None

        self._csp = ConversationSystemAndPermanent()

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
        from_pretrained_model_path = \
            self._application_paths.configuration_file_paths[
                "from_pretrained_model"]

        if from_pretrained_model_path.exists():
            from_pretrained_model_configuration = \
                FromPretrainedModelConfiguration.from_yaml(
                    from_pretrained_model_path)
        else:
            from_pretrained_model_configuration = \
                FromPretrainedModelConfiguration()

        from_pretrained_tokenizer_path = \
            self._application_paths.configuration_file_paths[
                "from_pretrained_tokenizer"]

        if from_pretrained_tokenizer_path.exists():
            from_pretrained_tokenizer_configuration = \
                FromPretrainedTokenizerConfiguration.from_yaml(
                    from_pretrained_tokenizer_path)
        else:
            from_pretrained_tokenizer_configuration = \
                FromPretrainedTokenizerConfiguration()

        generation_configuration_path = \
            self._application_paths.configuration_file_paths["generation"]

        if generation_configuration_path.exists():
            generation_configuration = \
                GenerationConfiguration.from_yaml(generation_configuration_path)
        else:
            generation_configuration = GenerationConfiguration()

        return (
            from_pretrained_model_configuration,
            from_pretrained_tokenizer_configuration,
            generation_configuration)

    def _load_model_and_tokenizer(
            self,
            model_path,
            from_pretrained_model_configuration = None,
            from_pretrained_tokenizer_configuration = None,
            generation_configuration = None):

        self._mat = ModelAndTokenizer(
            model_path=model_path,
            from_pretrained_model_configuration=from_pretrained_model_configuration,
            from_pretrained_tokenizer_configuration=from_pretrained_tokenizer_configuration,
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