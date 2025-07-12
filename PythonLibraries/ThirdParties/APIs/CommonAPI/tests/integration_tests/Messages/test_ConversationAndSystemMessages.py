
from commonapi.Messages import (
    AssistantMessage,
    ConversationAndSystemMessages,
    UserMessage)

from corecode.FileIO import JSONFile

def setup_conversation_and_system_messages():
    # https://prompts.chat/
    system_prompt = (
        "I want you to act as a prompt generator for Midjourney’s artificial "
        "intelligence program. Your job is to provide detailed and creative "
        "descriptions that will inspire unique and interesting images from the "
        "AI. Keep in mind that the AI is capable of understanding a wide range "
        "of language and can interpret abstract concepts, so feel free to be "
        "as imaginative and descriptive as possible. For example, you could "
        "describe a scene from a futuristic city, or a surreal landscape "
        "filled with strange creatures. The more detailed and imaginative your "
        "description, the more interesting the resulting image will be. Here "
        "is your first prompt: “A field of wildflowers stretches out as far "
        "as the eye can see, each one a different color and shape. In the "
        "distance, a massive tree towers over the landscape, its branches "
        "reaching up to the sky like tentacles.”"
    )

    conversation_and_system_messages = ConversationAndSystemMessages()
    conversation_and_system_messages.add_default_system_message()
    conversation_and_system_messages.add_system_message(system_prompt)

    conversation_and_system_messages.append_message(UserMessage(
        content=\
            (
                "Can you write a short introduction about the relevance of the "
                "term \"monopsony\" in economics? Please use examples related "
                "to potential monopsonies in the labour market and cite "
                "relevant research."
            )))
    conversation_and_system_messages.append_message(AssistantMessage(
        content=\
            (
                "The term \"monopsony\" refers to a market structure where a "
                "single buyer dominates the market, giving it significant "
                "influence over prices and conditions. In labour economics, "
                "monopsony power has gained attention for its potential to "
                "suppress wages and degrade working conditions. A classic "
                "example is tech giants like Google and Amazon in Silicon "
                "Valley, where they largely dictate salaries and benefits.\n\n"
                "Research suggests that monopsony power can lead to lower "
                "wages, reduced worker mobility, and decreased productivity. "
                "A study by Azar, Schmalz, and Tecu (2018) found that firms "
                "with significant monopsony power pay their workers 10-20% "
                "less than competitive firms. Similarly, a report by the "
                "Economic Policy Institute (2020) highlighted the growing "
                "concern of monopsony power in the US labour market, citing "
                "the concentration of employers in certain industries.\n\n"
                "Other examples include hospital systems in rural areas, "
                "where a single hospital may be the primary employer, giving "
                "it considerable bargaining power over nurses and other "
                "healthcare workers. The resulting low wages and poor working "
                "conditions can lead to staff shortages and decreased quality of care.\n\n"
                "The concept of monopsony power has implications for "
                "policymakers and regulators, as it challenges traditional "
                "views of labour markets and competition. By understanding "
                "the dynamics of monopsony power, researchers and "
                "policymakers can better address issues of wage stagnation, "
                "income inequality, and worker welfare.\n\n"
                "References:\n"
                "Azar, J., Schmalz, M. C., & Tecu, I. (2018). "
                "Labor market concentration. Journal of Human Resources, 53(4), 841-869.\n\n"
                "Economic Policy Institute. (2020). The growth of monopsony power in the US labor market."
            )))

    conversation_and_system_messages.append_message(UserMessage(
        content="Now explain it to a dog"))

    conversation_and_system_messages.append_message(AssistantMessage(
        content=\
            (
                "So, imagine you're at the dog park and there's only one guy "
                "with treats. All the dogs want treats, but they have to go to "
                "that one guy to get them. He gets to decide who gets a treat "
                "and how many they get. That's kinda like a monopsony, but "
                "instead of treats, it's a company that has a lot of power over "
                "the jobs.\n\nNow, imagine all the dog walkers in the neighborhood "
                "only work for that one guy. He gets to decide how much they get "
                "paid and what they have to do. That's not fair, right? The dog "
                "walkers might not get paid enough or get to walk as many dogs "
                "as they want.\n\nIt's similar with people who work for "
                "companies. If there's only one company that hires people with "
                "certain skills,, they might get to decide how much they're "
                "paid and what they have to do. That's called a monopsony, "
                "and it's not good for the workers.\n\nDoes that make sense, "
                "pup?")))
    return conversation_and_system_messages

more_user_message_0 = \
    "How can one fight back when a monospony had been created?"

more_assistant_message_0 = \
    (
        "Fighting back against a monopsony requires a multi-faceted approach. "
        "Here are some strategies:\n\n**Worker Organization**: When workers "
        "unite, they can gain bargaining power. For example, in 2020, Amazon "
        "warehouse workers in Alabama voted to unionize, which could help them "
        "negotiate better wages and working conditions. Collective bargaining "
        "agreements can help workers secure better wages, benefits, and "
        "working conditions.\n\n**Antitrust Litigation**: Governments can take "
        "action against companies that have created a monopsony. In the US, "
        "the Federal Trade Commission (FTC) and the Department of Justice "
        "(DOJ) have launched investigations into companies like Google, "
        "Amazon, and Facebook for potential antitrust violations. For "
        "instance, the FTC's lawsuit against Facebook in 2020 alleged that "
        "the company had acquired Instagram and WhatsApp to eliminate "
        "competition.\n\n**Regulatory Action**: Policymakers can create laws "
        "and regulations to prevent monopsony power. For example, the EU's "
        "Digital Markets Act aims to regulate the activities of large tech "
        "companies and prevent them from abusing their market power. In the "
        "US, lawmakers have proposed bills like the \"Monopsony Prevention Act\" "
        "to address labor market concentration.\n\n**Promoting Competition**: "
        "Encouraging new entrants into the market can help break up a monopsony. "
        "This can be achieved through policies like tax incentives, subsidies, "
        "or investment in infrastructure. For example, the US government's "
        "efforts to promote rural broadband development aim to increase "
        "competition in areas where a single provider may dominate.\n\n"
        "**Worker Mobility**: Improving worker mobility can help reduce the "
        "power of a monopsony. This can be achieved through policies like "
        "portable benefits, which allow workers to take their benefits with "
        "them if they change jobs. A study by the Aspen Institute found that "
        "portable benefits could increase worker mobility and reduce the "
        "negative effects of monopsony power.\n\n**Public Awareness**: Raising "
        "awareness about the issues caused by monopsony power can help build "
        "public support for action. Organizations like the Economic Policy "
        "Institute and the Center for American Progress have published research "
        "and analysis on the topic, highlighting the need for policymakers to "
        "address monopsony power.\n\n**Support for Alternative Models**: "
        "Alternative models like worker cooperatives, where workers own and "
        "control the business, can provide a more equitable alternative to "
        "traditional corporate structures. For example, the Mondragon "
        "Cooperative Corporation in Spain is a successful example of a "
        "worker-owned cooperative that has thrived in a competitive market.\n\n"
        "It's often a combination of these strategies that can effectively "
        "challenge a monopsony. What's the specific context you're interested in?"
    )

def test_clear_conversation_history_can_keep_active_system_messages():
    conversation_and_system_messages = setup_conversation_and_system_messages()
    assert len(
        conversation_and_system_messages.get_conversation_as_list_of_dicts()) \
            == 6
    conversation_and_system_messages.clear_conversation_history()
    list_of_messages_dict = \
        conversation_and_system_messages.get_conversation_as_list_of_dicts()
    assert len(list_of_messages_dict) == 2
    assert list_of_messages_dict[0]["role"] == "system"

    original_conversation_and_system_messages = \
        setup_conversation_and_system_messages()
    assert list_of_messages_dict[0]["content"] == \
        original_conversation_and_system_messages.system_messages_manager.messages[0].content
    assert list_of_messages_dict[1]["content"] == \
        original_conversation_and_system_messages.system_messages_manager.messages[1].content

    conversation_and_system_messages.append_message(UserMessage(
        content=more_user_message_0))
    conversation_and_system_messages.append_message(AssistantMessage(
        content=more_assistant_message_0))
    list_of_messages_dict = \
        conversation_and_system_messages.get_conversation_as_list_of_dicts()
    assert len(list_of_messages_dict) == 4
    assert list_of_messages_dict[2]["content"] == more_user_message_0
    assert list_of_messages_dict[3]["content"] == more_assistant_message_0

    conversation_and_system_messages.clear_conversation_history()
    list_of_messages_dict = \
        conversation_and_system_messages.get_conversation_as_list_of_dicts()
    assert len(list_of_messages_dict) == 2
    assert list_of_messages_dict[0]["content"] == \
        original_conversation_and_system_messages.system_messages_manager.messages[0].content
    assert list_of_messages_dict[1]["content"] == \
        original_conversation_and_system_messages.system_messages_manager.messages[1].content

