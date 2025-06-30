from moregroq.Wrappers.GetAllActiveModels import GetAllActiveModels

class ModelSelector:
    def __init__(self, api_key: str):
        self._get_all_active_models = GetAllActiveModels(api_key)
        self._get_all_active_models()
        self.available_models = \
            self._get_all_active_models.get_list_of_available_models()

        # model "id" would better be described as name or "model_name"
        self._model_name_to_index_dict = {}
        for index, model in enumerate(self.available_models):
            self._model_name_to_index_dict[model["id"]] = index

        self.current_model = None

    def get_abbreviated_model_data(self, model_name: str):
        return self.available_models[self._model_name_to_index_dict[model_name]]

    def get_all_available_models(self):
        return self.available_models

    def set_current_model_by_index(self, index: int):
        self.current_model = self.available_models[index]

    def get_context_window_by_model_name(self, model_name: str):
        return int(
            self.available_models[
                self._model_name_to_index_dict[model_name]]["context_window"])