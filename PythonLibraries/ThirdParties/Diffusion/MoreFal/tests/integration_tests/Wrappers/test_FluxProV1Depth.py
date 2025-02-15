from corecode.Utilities import load_environment_file
from morefal.Wrappers import FluxProV1Depth

import pytest

load_environment_file()

@pytest.mark.asyncio
async def test_FluxProV1Depth_can_generate():
    generator = FluxProV1Depth()
    prompt = (
        "75 years later: Aerial wide shot of a dilapidated New England "
        "sprawling estate on a lake, late afternoon, with overgrown gardens "
        "on a terrace near the water after decades of decline."
    )
    image_url = "https://v3.fal.media/files/zebra/QJQrTXBwUv0T73VQnZkft_Estate_001_1930s_jpg.jpg"
    
    result = await generator.generate(prompt, image_url)
    
    assert result is not None
    assert result.url.startswith("https://")
    assert result.width > 0
    assert result.height > 0

    print(result)