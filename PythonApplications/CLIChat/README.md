# List of possible "free" API keys

* groq - [Groq cloud api and model](https://console.groq.com/keys) / [Groq Hosted models](https://console.groq.com/docs/models)

* mistral - Mistral AI (recent addition)

## Groq Cloud API Key

Groq Cloud API Key allows you to have FREE access to [selected open source LLMs](https://console.groq.com/docs/models).

At the time of writing, use of Groq Cloud API is FREE.

### Generate Groq API Key

1. Go to https://console.groq.com/keys
2. Log in with a registered account
3. Click menu item "API Keys" on the left
4. Click button "Create API Key"
5. Enter a name, for example, "clichat"
6. Copy or make a note of the created API key

## Mistral AI API Key Setup

Mistral AI API Key allows you to have FREE access to [selected open source LLMs](https://docs.mistral.ai/getting-started/models/models_overview/).

At the time of writing, Mistral AI offers API keys for both FREE and paid tier users.

Even FREE tier users can use Mistral Large models

### Generate Mistral API Key

![api_setup](https://github.com/user-attachments/assets/a93d6875-dbe8-44d6-84d4-6f924e6d54aa)

1. Go to https://console.mistral.ai/api-keys/
2. Log in with a registered account (Note that each free plan requires a phone number to verify.)
3. Click menu item "API Keys" on the left
4. Click button "Create new Key"
5. Enter a name, for example, "toolmate"
6. Copy or make a note of the created API key

# Deployment

Make sure, if you made changes either to the `ThirdParties/MoreGroq` Python library/code or to the `CLIChat` Python application, that you run `poetry build` in each respective directory where there's a pyproject.toml file in order to build a new .tar.gz to place into the `clichat-installer` subdirectory.

Run

```
pyinstaller installer.spec
```
in the subdirectory containing `installer.spec`, which is in `Deployment/`

# Development Scratch notes

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
