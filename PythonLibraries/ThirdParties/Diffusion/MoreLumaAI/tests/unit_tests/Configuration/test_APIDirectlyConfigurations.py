from morelumaai.Configuration.APIDirectlyConfigurations import (
    Ray2Configuration)

def test_Ray2Configuration_inits_with_defaults():
    configuration = Ray2Configuration()
    assert configuration.model == "ray-2"
    assert configuration.aspect_ratio == None
    assert configuration.fps == None
    assert configuration.duration == None
    assert configuration.loop == None
    api_kwargs = configuration.to_api_kwargs()
    assert api_kwargs == {"model": "ray-2"}

