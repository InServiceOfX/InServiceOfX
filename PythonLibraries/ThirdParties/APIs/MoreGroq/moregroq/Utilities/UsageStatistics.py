class UsageStatistics:
    @staticmethod
    def _parse_groq_response(response):
        """
        Args: response: ChatCompletion
        ChatCompletion is a class in the groq API, i.e.
        <class 'groq.types.chat.chat_completion.ChatCompletion'>
        """
        single_statistic = {
            "model": response.model,
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
            "completion_time": response.usage.completion_time,
            "prompt_time": response.usage.prompt_time,
            "queue_time": response.usage.queue_time,
            "total_time": response.usage.total_time,
        }
        return single_statistic
