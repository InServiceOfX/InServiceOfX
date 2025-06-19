from morepydanticai.RAG import parse_online_json
from morepydanticai.RAG.LogfireExample import create_type_adapter, DocsSection

from TestSetup.TestData import pydantic_ai_rag_test_data

import pytest

@pytest.mark.asyncio
async def test_parse_online_json():
    url = pydantic_ai_rag_test_data()[2]
    type_adapter = create_type_adapter()

    sections = await parse_online_json(url, type_adapter)
    assert sections is not None
    assert len(sections) == 299
    assert isinstance(sections[0], DocsSection)

    total_length = 0
    for section in sections:
        total_length += len(section.content)

    assert total_length == 182962