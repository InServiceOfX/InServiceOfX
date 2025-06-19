import httpx

from pydantic import TypeAdapter

async def parse_online_json(url: str, type_adapter: TypeAdapter):

    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
    sections = type_adapter.validate_json(response.content)
    return sections