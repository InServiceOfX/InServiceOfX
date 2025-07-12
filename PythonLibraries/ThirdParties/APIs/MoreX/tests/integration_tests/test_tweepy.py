from corecode.FileIO import TextFile
from datetime import datetime
from morex.Configuration import LoadMoreXEnvironment
from morex.Utilities import parse_post_URL
from pathlib import Path

import tweepy

def test_tweepy_with_only_bearer_token():
    post_URL = "https://x.com/alex_prompter/status/1943232047738425642"
    _, post_id = parse_post_URL(post_URL)

    load_morex_environment = LoadMoreXEnvironment()
    assert load_morex_environment._path_to_env_file == \
        Path(__file__).parents[6] / "Configurations" / "ThirdParties" / \
        "APIs" / "MoreX" / "morex.env"
    load_morex_environment()

    client = tweepy.Client(
        bearer_token=load_morex_environment.get_environment_variable(
            "X_BEARER_TOKEN"
        )
    )

    response = client.get_tweet(id=post_id)
    print(response)
    assert isinstance(response, tweepy.tweet.Tweet)
    assert response.id == post_id
    assert response.includes == {}
    assert response.errors == []
    assert response.meta == {}
    print(response.text)

def test_tweepy_with_developer_account():
    post_URL = "https://x.com/alex_prompter/status/1943232047738425642"
    _, post_id = parse_post_URL(post_URL)

    load_morex_environment = LoadMoreXEnvironment()
    load_morex_environment()

    client = tweepy.Client(
        consumer_key=load_morex_environment.get_environment_variable(
            "X_CONSUMER_API_KEY"
        ),
        consumer_secret=load_morex_environment.get_environment_variable(
            "X_CONSUMER_SECRET"
        ),
        access_token=load_morex_environment.get_environment_variable(
            "X_ACCESS_TOKEN"
        ),
        access_token_secret=load_morex_environment.get_environment_variable(
            "X_SECRET_TOKEN"
        )
    )

    response = client.get_tweet(id=post_id)
    assert isinstance(response, tweepy.tweet.Tweet)
    assert response.id == post_id
    assert response.includes == {}
    assert response.errors == []
    assert response.meta == {}
    print(response.text)

def test_tweepy_for_OAuth1_0a():
    post_URL = "https://x.com/alex_prompter/status/1943231978779877514"
    _, post_id = parse_post_URL(post_URL)

    load_morex_environment = LoadMoreXEnvironment()
    load_morex_environment()

    consumer_key = load_morex_environment.get_environment_variable(
        "X_CONSUMER_API_KEY"
    )
    consumer_secret = load_morex_environment.get_environment_variable(
        "X_CONSUMER_SECRET"
    )
    access_token = load_morex_environment.get_environment_variable(
        "X_ACCESS_TOKEN"
    )
    access_token_secret = load_morex_environment.get_environment_variable(
        "X_SECRET_TOKEN"
    )

    auth = tweepy.OAuth1UserHandler(
        consumer_key,
        consumer_secret,
        access_token,
        access_token_secret,
    )

    api = tweepy.API(auth, wait_on_rate_limit=True)

    tweet = api.get_status(post_id, tweet_mode="extended")

    assert tweet.user.screen_name == "alex_prompter"
    assert tweet.created_at == datetime(2025, 7, 10, 12, 0, 0)

def test_tweepy_with_bearer_token_and_wait():
    post_URL = "https://x.com/alex_prompter/status/1943231978779877514"
    _, post_id = parse_post_URL(post_URL)

    load_morex_environment = LoadMoreXEnvironment()
    load_morex_environment()

    client = tweepy.Client(
        bearer_token=load_morex_environment.get_environment_variable(
            "X_BEARER_TOKEN"
        ),
        wait_on_rate_limit=True
    )

    response = client.get_tweet(
        id=post_id,
        tweet_fields=["author_id", "created_at", "text"]
    )
    tweet = response.data
    print(tweet)
    assert isinstance(tweet, tweepy.tweet.Tweet)
    assert tweet.id == post_id
    assert tweet.author_id == "1943231978779877514"
    assert tweet.created_at == datetime(2025, 7, 10, 12, 0, 0)
    print(tweet.text)

def test_fetching_only_author_replies():
    post_URL = "https://x.com/alex_prompter/status/1943231978779877514"
    author_name, post_id = parse_post_URL(post_URL)

    query = f"conversation_id:{post_id} from:{author_name}"
    replies = []

    pagination_token = None

    load_morex_environment = LoadMoreXEnvironment()
    load_morex_environment()

    client = tweepy.Client(
        bearer_token=load_morex_environment.get_environment_variable(
            "X_BEARER_TOKEN"
        ),
        wait_on_rate_limit=True
    )

    # ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_construct_response', '_get_authenticating_user_id', '_get_oauth_1_authenticating_user_id', '_get_oauth_2_authenticating_user_id', '_make_request', '_process_data', '_process_includes', '_process_params', 'access_token', 'access_token_secret', 'add_list_member', 'bearer_token', 'bookmark', 'consumer_key', 'consumer_secret', 'create_compliance_job', 'create_direct_message', 'create_direct_message_conversation', 'create_dm', 'create_dm_conversation', 'create_list', 'create_tweet', 'delete_list', 'delete_tweet', 'follow', 'follow_list', 'follow_user', 'get_all_tweets_count', 'get_blocked', 'get_bookmarks', 'get_compliance_job', 'get_compliance_jobs', 'get_direct_message_events', 'get_dm_events', 'get_followed_lists', 'get_home_timeline', 'get_liked_tweets', 'get_liking_users', 'get_list', 'get_list_followers', 'get_list_members', 'get_list_memberships', 'get_list_tweets', 'get_me', 'get_muted', 'get_owned_lists', 'get_pinned_lists', 'get_quote_tweets', 'get_recent_tweets_count', 'get_retweeters', 'get_space', 'get_space_buyers', 'get_space_tweets', 'get_spaces', 'get_tweet', 'get_tweets', 'get_user', 'get_users', 'get_users_followers', 'get_users_following', 'get_users_mentions', 'get_users_tweets', 'hide_reply', 'like', 'mute', 'pin_list', 'remove_bookmark', 'remove_list_member', 'request', 'return_type', 'retweet', 'search_all_tweets', 'search_recent_tweets', 'search_spaces', 'session', 'unfollow', 'unfollow_list', 'unfollow_user', 'unhide_reply', 'unlike', 'unmute', 'unpin_list', 'unretweet', 'update_list', 'user_agent', 'wait_on_rate_limit']
    #print(dir(client))

    while True:
        response = client.search_recent_tweets(
            query=query,
            tweet_fields=["created_at", "text"],
            max_results=100,
            next_token=pagination_token
        )
        print("Got response: ", response)
        replies.extend(response.data or [])
        pagination_token = response.meta.get("next_token")
        if pagination_token is None:
            break

    text_only_replies = []

    for tweet in replies:
        print("-", tweet.created_at, tweet.text)
        text_only_replies.append(tweet.text)

    file_path = Path.cwd() / "alex_prompter_replies.txt"
    TextFile.save_lines(file_path, text_only_replies)



