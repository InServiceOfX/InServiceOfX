from morepydanticai.RAG.LogfireExample import slugify

def test_slugify_ascii():
    # Basic ASCII, spaces and punctuation
    assert slugify("Hello, World!", "-") == "hello-world"
    assert slugify("Python's slugify function.", "_") == \
        "pythons_slugify_function"

def test_slugify_unicode():
    # žlutý kůň = yellow horse
    # Unicode characters should be converted to ASCII if unicode=False
    assert slugify("žlutý kůň", "-", unicode=False) == "zluty-kun"
    # If unicode=True, keep unicode characters
    assert slugify("žlutý kůň", "-", unicode=True) == "žlutý-kůň"