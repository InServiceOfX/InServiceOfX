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
