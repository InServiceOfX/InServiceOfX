Chatbot.py
class Chatbot

self.messages
* can be reset to just the system message.
* for each prompt result, gets added with it as a user message.
* gets passed as message to Groq API call,
* response from Groq API call is added as an "assistant" message, via
* the .new_chat_response of a configuration.

currentMessages
* in the StreamingWordWrapper, gets appended with the chat response as an "assistant" message.
* in the main loop of Chatbot, gets appended with the prompt as a user message.

new_chat_response
* new_chat_response is the response from the Groq API call and is copied or appended into self.messages in the main loop.
* it is written over each time in StreamingWordWrapper given a new chat_response.

## Refactor message creation and message persistence

### System messages

I want
* checklist of system messages to include at the start of messages.
- this is entered at beginning of session and when resetting system messages.
* can save to runtime list of system messages at any time when user wants to create a new system message.
- show existing system messages in a list.
* choose to save system messages to a file
- each system message assigned a sha256 hash, and timestamp of when saved.
