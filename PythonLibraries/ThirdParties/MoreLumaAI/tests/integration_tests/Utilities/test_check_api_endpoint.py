from morelumaai.Utilities import check_api_endpoint

def test_check_api_endpoint():
    url1 = "https://api.lumalabs.ai/dream-machine/v1/generations"
    is_reachable, error = check_api_endpoint(url1)
    assert is_reachable == True
    assert error == None
