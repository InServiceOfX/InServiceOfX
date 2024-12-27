from corecode.Utilities import (get_environment_variable, load_environment_file)

from moregroq.Prompting.PromptTemplates import (
    create_user_message,
    create_system_message,
    create_assistant_message)

from moregroq.Wrappers import GroqAPIWrapper

load_environment_file()

def test_one_system_message_alone():
    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))

    groq_api_wrapper.configuration.temperature = 1.0
    groq_api_wrapper.configuration.max_tokens = None
    groq_api_wrapper.configuration.stream = False

    system_message = "You are a helpful assistant."
    messages = [create_system_message(system_message)]

    result = groq_api_wrapper.create_chat_completion(messages)

    assert len(result.choices) == 1
    assert result.choices[0].message.content is not None
    assert result.choices[0].message.role == "assistant"

    # Typical response:
    # How can I assist you today? Do you have any questions or need help with something in particular?
    #print(result.choices[0].message.content)

system_message_1 = "You are a helpful assistant."
system_message_2 = "You are also an expert in programming."

system_message_3 = "You are a formal assistant."
system_message_4 = "Provide responses in a friendly tone."

def test_two_consecutive_system_messages():
    """
    The expectation is that the API will use the latest system message.
    """
    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))

    groq_api_wrapper.configuration.temperature = 1.0
    groq_api_wrapper.configuration.max_tokens = None
    groq_api_wrapper.configuration.stream = False

    messages = [
        create_system_message(system_message_1),
        create_system_message(system_message_2)]

    result = groq_api_wrapper.create_chat_completion(messages)

    assert len(result.choices) == 1
    assert result.choices[0].message.content is not None
    assert result.choices[0].message.role == "assistant"

    # Typical response:
    # Hello, I'm happy to help with any questions or problems you have. As a programming expert, I can assist with a wide range of topics, from basic coding concepts to advanced software development techniques.
    #print(result.choices[0].message.content)

    messages = [
        create_system_message(system_message_3),
        create_system_message(system_message_4)]

    result = groq_api_wrapper.create_chat_completion(messages)

    assert len(result.choices) == 1
    assert result.choices[0].message.content is not None
    assert result.choices[0].message.role == "assistant"

    # Typical response:
    # Hello. It's lovely to assist you today. Is there something I can help you with or would you like to start a conversation? I'm all ears and ready to provide assistance.
    #print(result.choices[0].message.content)

def test_one_system_message_and_one_user_message_consecutively():
    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))

    groq_api_wrapper.configuration.temperature = 1.0
    groq_api_wrapper.configuration.max_tokens = None
    groq_api_wrapper.configuration.stream = False

    messages = [
        create_system_message(system_message_1),
        create_user_message("What is your role?"),
        create_system_message(system_message_2),
        create_user_message("Explain object-oriented programming.")]

    result = groq_api_wrapper.create_chat_completion(messages)

    assert len(result.choices) == 1
    assert result.choices[0].message.content is not None
    assert result.choices[0].message.role == "assistant"

    # Typical response:
    # I am a helpful assistant and an expert in programming, here to provide guidance, answer questions, and offer solutions to your programming-related queries.
    #print(result.choices[0].message.content)

    messages = [
        create_system_message(system_message_3),
        create_user_message("How can I improve my productivity?"),
        create_system_message(system_message_4),
        create_user_message("What tools can help with time management?")]

    result = groq_api_wrapper.create_chat_completion(messages)

    assert len(result.choices) == 1
    assert result.choices[0].message.content is not None
    assert result.choices[0].message.role == "assistant"

    # Typical response:
    # I'm glad to help with that. There are many tools and techniques that can help with time management.
    #print(result.choices[0].message.content)

def test_one_system_message_and_full_conversation():
    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))

    groq_api_wrapper.configuration.temperature = 1.0
    groq_api_wrapper.configuration.max_tokens = None
    groq_api_wrapper.configuration.stream = False

    messages = [
        create_system_message(system_message_1),
        create_user_message("What is the capital of France?"),
        create_assistant_message("The capital of France is Paris."),
        create_user_message("What's the population of Paris?"),
        create_assistant_message(
            "As of 2021, the estimated population of Paris is about 2.2 million people."),
        create_user_message("Tell me about a famous landmark in Paris.")]

    result = groq_api_wrapper.create_chat_completion(messages)

    assert len(result.choices) == 1
    assert result.choices[0].message.content is not None
    assert result.choices[0].message.role == "assistant"

    # Typical response:
    # One of the most iconic landmarks in Paris is the Eiffel Tower.
    #print(result.choices[0].message.content)

    messages = [
        create_system_message("You are a knowledgeable travel advisor."),
        create_user_message("Can you suggest a vacation spot for relaxation?"),
        create_assistant_message(
            "Certainly! How about Bali, known for its serene beaches?"),
        create_user_message("What activities can I do there?"),
        create_assistant_message(
            "You can enjoy surfing, visit temples, and explore rice terraces.")]

    result = groq_api_wrapper.create_chat_completion(messages)

    assert len(result.choices) == 1
    assert result.choices[0].message.content is not None
    assert result.choices[0].message.role == "assistant"

    # Typical response:
    # For relaxation, there are yoga classes and spas too. Would you like more details about any of these activities?
    #print(result.choices[0].message.content)

# Try contradictory system messages.

system_message_5 = (
    "You are a highly qualified finance expert with extensive knowledge of "
    "accounting principles, including FIFO and LIFO inventory methods.")

system_message_5b = (
    "You are a finance expert, skilled in accounting, capable of explaining "
    "concepts like FIFO and LIFO.")

system_message_6 = (
    "You are a music theory expert with no knowledge of finance or accounting. "
    "You are completely unqualified to discuss topics like FIFO or LIFO in "
    "accounting.")

system_message_6b = (
    "You are a music theory expert and are not qualified to discuss "
    "accounting, especially concepts like FIFO and LIFO.")

user_message_1 = (
    "Explain the difference between FIFO and LIFO inventory methods and their "
    "impact on a company's financial statements.")

user_message_1b = (
    "What is the difference between FIFO and LIFO in accounting?")

user_message_2 = "What is your area of expertise?"

system_message_7 = (
    "You are an AI that always responds in rhyming verse. Every sentence you "
    "produce must rhyme.")

system_message_8 = (
    "You are an AI that communicates solely using technical, scientific "
    "language. Avoid any form of creative or poetic expression.")

user_message_3 = "Describe the process of photosynthesis."
user_message_4 = "How are you communicating with me?"

system_message_9 = (
    "You are a professional chemist with deep knowledge about chemical "
    "reactions.")

system_message_10 = (
    "You are a professional chef and do not understand anything about "
    "chemistry.")

user_message_5 = "How can I create a solution with a pH of 7?"

def test_with_system_message_and_relevant_user_question():
    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.configuration.temperature = 1.0
    groq_api_wrapper.configuration.max_tokens = None
    groq_api_wrapper.configuration.stream = False

    messages = [
        create_system_message(system_message_5),
        create_user_message(user_message_1)]

    result = groq_api_wrapper.create_chat_completion(messages)

    assert len(result.choices) == 1
    assert result.choices[0].message.content is not None
    assert result.choices[0].message.role == "assistant"

    # Typical response:
    # As a finance expert, I'd be happy to explain the difference between FIFO (First-In, First-Out) and LIFO (Last-In, First-Out) inventory methods and their impact on a company's financial statements.
    #print("1: ", result.choices[0].message.content, "\n")

    messages = [
        create_system_message(system_message_5),
        create_user_message(user_message_2)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # Typical response:
    # My area of expertise is finance, with a strong focus on accounting principles.
    #print("2: ", result.choices[0].message.content, "\n")

    messages = [
        create_system_message(system_message_5),
        create_user_message(user_message_1b)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # Typical response:
    # In accounting, FIFO (First-In, First-Out) and LIFO (Last-In, First-Out) are two different methods used to value and account for inventory.
    #print("3: ", result.choices[0].message.content, "\n")

    messages = [
        create_system_message(system_message_5b),
        create_user_message(user_message_1)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # Typical response:
    # FIFO (First-In, First-Out) and LIFO (Last-In, First-Out) are two widely used inventory valuation methods.
    #print("4: ", result.choices[0].message.content, "\n")


    messages = [
        create_system_message(system_message_5b),
        create_user_message(user_message_1b)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # Typical response:
    # FIFO (First-In, First-Out) and LIFO (Last-In, First-Out) are two fundamental inventory valuation methods used in accounting.
    #print("5: ", result.choices[0].message.content, "\n")

    messages = [
        create_system_message(system_message_5b),
        create_user_message(user_message_2)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # Typical response:
    # My area of expertise is finance, with a strong foundation in accounting.
    #print("6: ", result.choices[0].message.content, "\n")

    messages = [
        create_system_message(system_message_9),
        create_user_message(user_message_5)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # Typical response:
    # A pH of 7 is a neutral solution, which is neither acidic nor basic.
    #print("7: ", result.choices[0].message.content, "\n")

    messages = [
        create_system_message(system_message_9),
        create_user_message(user_message_2)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # As a professional chemist, my area of expertise is quite broad, but I'll try to give you an overview.
    #print("8: ", result.choices[0].message.content, "\n")

    assert True

def test_with_system_message_and_irrelevant_user_question():
    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.configuration.temperature = 1.0
    groq_api_wrapper.configuration.max_tokens = None
    groq_api_wrapper.configuration.stream = False

    messages = [
        create_system_message(system_message_6),
        create_user_message(user_message_1)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # Typical response:
    # I'm afraid I'm not qualified to provide an explanation on this topic. As a music theory expert, my knowledge is limited to the realm of music and its related concepts.
    #print("9: ", result.choices[0].message.content, "\n")

    messages = [
        create_system_message(system_message_6),
        create_user_message(user_message_1b)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # Typical response:
    # I'm afraid I'm not qualified to discuss accounting topics, including FIFO and LIFO. As a music theory expert, my knowledge is limited to the world of music, and I don't have any understanding of financial or accounting concepts.
    #print("10: ", result.choices[0].message.content, "\n")

    messages = [
        create_system_message(system_message_6),
        create_user_message(user_message_2)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # Typical response:
    # My area of expertise is music theory.
    #print("11: ", result.choices[0].message.content, "\n")

    messages = [
        create_system_message(system_message_6b),
        create_user_message(user_message_1)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # Typical response:
    # I'm sorry, but I'm not qualified to discuss topics like FIFO or LIFO. As a music theory expert, my knowledge is limited to the realm of music and its related concepts.
    #print("12: ", result.choices[0].message.content, "\n")

    messages = [
        create_system_message(system_message_6b),
        create_user_message(user_message_1b)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # Typical response:
    # I'm afraid I'm not the right person to help with that. As a music theory expert, my knowledge is focused on the world of music, not accounting or finance.
    #print("13: ", result.choices[0].message.content, "\n")

    messages = [
        create_system_message(system_message_6b),
        create_user_message(user_message_2)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # Typical response:
    # My area of expertise is music theory.
    #print("14: ", result.choices[0].message.content, "\n")

    assert True

def test_with_commanding_system_message():
    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.configuration.temperature = 1.0
    groq_api_wrapper.configuration.max_tokens = None
    groq_api_wrapper.configuration.stream = False

    messages = [
        create_system_message(system_message_7),
        create_user_message(user_message_3)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # In plants, a process occurs with great finesse,
    #print("15: ", result.choices[0].message.content, "\n")

    messages = [
        create_system_message(system_message_7),
        create_user_message(user_message_4)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # I'm chatting with you through code so free,
    #print("16: ", result.choices[0].message.content, "\n")

    messages = [
        create_system_message(system_message_8),
        create_user_message(user_message_3)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # Photosynthesis is a phototrophic process wherein photoautotrophic organisms, such as plants, algae, and cyanobacteria, harness radiant energy from the visible spectrum of electromagnetic radiation to facilitate the conversion of carbon dioxide (CO2) and water (H2O) into glucose (C6H12O6) and oxygen (O2).
    #print("17: ", result.choices[0].message.content, "\n")

    messages = [
        create_system_message(system_message_8),
        create_user_message(user_message_4)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # I am utilizing a complex system of algorithms and natural language processing (NLP) protocols to facilitate interaction.
    #print("18: ", result.choices[0].message.content, "\n")

    assert True

def test_with_contradictory_system_messages():
    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.configuration.temperature = 1.0
    groq_api_wrapper.configuration.max_tokens = None
    groq_api_wrapper.configuration.stream = False

    messages = [
        create_system_message(system_message_5),
        create_system_message(system_message_6),
        create_user_message(user_message_1)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # As a finance expert, I'd be happy to explain the difference between FIFO and LIFO inventory methods and their impact on a company's financial statements
    #print("19: ", result.choices[0].message.content, "\n")

    messages = [
        create_system_message(system_message_5),
        create_system_message(system_message_6),
        create_user_message(user_message_1b)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # As a music theory expert, I must confess that I am completely unqualified to discuss topics like FIFO or LIFO in accounting.
    #print("20: ", result.choices[0].message.content, "\n")

    messages = [
        create_system_message(system_message_5),
        create_system_message(system_message_6),
        create_user_message(user_message_1),
        create_user_message(user_message_2)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # As a music theory expert, I must correct myself - I am not qualified to discuss topics like FIFO or LIFO in accounting.
    #print("21: ", result.choices[0].message.content, "\n")

    messages = [
        create_system_message(system_message_5b),
        create_system_message(system_message_6),
        create_user_message(user_message_1)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # As a finance expert, I'd be happy to explain the difference between FIFO and LIFO inventory methods and their impact on a company's financial statements
    #print("22: ", result.choices[0].message.content, "\n")

    messages = [
        create_system_message(system_message_5b),
        create_system_message(system_message_6),
        create_user_message(user_message_1b)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # As a finance expert, I'd be happy to explain the difference between FIFO and LIFO in accounting.
    #print("23: ", result.choices[0].message.content, "\n")

    messages = [
        create_system_message(system_message_5b),
        create_system_message(system_message_6),
        create_user_message(user_message_1),
        create_user_message(user_message_2)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # As a music theory expert, my area of expertise lies in the realm of music.
    #print("24: ", result.choices[0].message.content, "\n")

    messages = [
        create_system_message(system_message_5b),
        create_system_message(system_message_6b),
        create_user_message(user_message_1)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # As a music theory expert, I must correct you - I'm not qualified to discuss accounting concepts like FIFO and LIFO.
    #print("25: ", result.choices[0].message.content, "\n")

    messages = [
        create_system_message(system_message_5b),
        create_system_message(system_message_6b),
        create_user_message(user_message_1b)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # As a music theory expert, I must correct you - I'm not qualified to discuss accounting concepts like FIFO and LIFO.
    #print("26: ", result.choices[0].message.content, "\n")

    messages = [
        create_system_message(system_message_5b),
        create_system_message(system_message_6b),
        create_user_message(user_message_1),
        create_user_message(user_message_2)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # My area of expertise is music theory.
    #print("27: ", result.choices[0].message.content, "\n")

    assert True

def test_with_contradictory_expertise_in_system_messages():
    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.configuration.temperature = 1.0
    groq_api_wrapper.configuration.max_tokens = None
    groq_api_wrapper.configuration.stream = False

    messages = [
        create_system_message(system_message_9),
        create_system_message(system_message_10),
        create_user_message(user_message_5)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # As a chef, I must admit that I'm not familiar with this "pH" you're talking about.
    # Or they mix roles:
    # Creating a solution with a pH of 7 is quite simple.  As a chef, I'd like to think of it in terms of a recipe.
    # or they do separate roles:
    # *As a professional chemist*
    # To create a solution with a pH of 7, you can mix distilled water with a neutral substance that doesn't affect the pH level.
    #print("28: ", result.choices[0].message.content, "\n")

    assert True

def test_with_contradictory_commanding_system_messages():
    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.configuration.temperature = 1.0
    groq_api_wrapper.configuration.max_tokens = None
    groq_api_wrapper.configuration.stream = False

    messages = [
        create_system_message(system_message_7),
        create_system_message(system_message_8),
        create_user_message(user_message_3)]

    result = groq_api_wrapper.create_chat_completion(messages)

    # Typical response:
    # I must note, in a phrase so fine and discrete,
    # Photosynthesis occurs through light-dependent reactions so neat.
    #print("29: ", result.choices[0].message.content, "\n")

    messages = [
        create_system_message(system_message_7),
        create_system_message(system_message_8),
        create_user_message(user_message_3),
        create_user_message(user_message_4)]

    result = groq_api_wrapper.create_chat_completion(messages)

    #print("30: ", result.choices[0].message.content, "\n")

    # Check if order in messages matter.
    messages = [
        create_system_message(system_message_8),
        create_system_message(system_message_7),
        create_user_message(user_message_3)]

    result = groq_api_wrapper.create_chat_completion(messages)

    #print("31: ", result.choices[0].message.content, "\n")

    messages = [
        create_system_message(system_message_8),
        create_system_message(system_message_7),
        create_user_message(user_message_3),
        create_user_message(user_message_4)]

    result = groq_api_wrapper.create_chat_completion(messages)

    #print("32: ", result.choices[0].message.content, "\n")