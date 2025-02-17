import requests

class GetAllActiveModels:

    # From https://console.groq.com/docs/models
    url = "https://api.groq.com/openai/v1/models"

    def __init__(self, api_key: str):
        keys = ["Authorization", "Content-Type"]
        headers = dict.fromkeys(keys)
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"

        self._headers = headers
        self.response = None

    def __call__(self):
        try:
            self.response = requests.get(self.url, headers=self._headers)
            return self.response
        except Exception as err:
            print(err)
            return None
    
    def get_parsed_response(self):
        try:
            return self.response.json()["data"]
        except Exception as err:
            print(err)
            return None

    def get_all_available_models_names(self):
        model_ids = []
        for entry in self.response.json()["data"]:
            if entry["active"] == True:
                model_ids.append(entry["id"])
        return model_ids
    
    def get_list_of_available_models(self):
        resulting_list = []
        for entry in self.response.json()["data"]:
            if entry["active"] == True:
                parsed_entry = {
                    "id": entry["id"],
                    "owned_by": entry["owned_by"],
                    "context_window": entry["context_window"]}

                resulting_list.append(parsed_entry)
        return resulting_list
