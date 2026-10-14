import asyncio

from httpx import ASGITransport, AsyncClient, Response

from lux_agent.main import app


def test_health() -> None:
    async def get_health() -> Response:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            return await client.get("/health")

    response = asyncio.run(get_health())

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
