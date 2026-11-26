from .helpers import KEY, api, make_services


async def test_without_a_configured_key_auth_is_off() -> None:
    async with api(make_services()) as client:
        assert (await client.get("/demo/cases")).status_code == 200


async def test_a_missing_key_is_refused() -> None:
    async with api(make_services(api_key=KEY)) as client:
        response = await client.get("/demo/cases")
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "ApiKey"


async def test_a_wrong_key_is_refused_with_the_same_answer() -> None:
    async with api(make_services(api_key=KEY)) as client:
        missing = await client.get("/documents/doc-1")
    async with api(make_services(api_key=KEY), key="wrong") as client:
        wrong = await client.get("/documents/doc-1")
    assert wrong.status_code == missing.status_code == 401
    assert wrong.json() == missing.json()


async def test_the_right_key_gets_through() -> None:
    async with api(make_services(api_key=KEY), key=KEY) as client:
        assert (await client.get("/documents/doc-1")).status_code == 404


async def test_health_needs_no_key() -> None:
    async with api(make_services(api_key=KEY)) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
