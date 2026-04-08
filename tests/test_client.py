"""Tests for Client and ChatClient Protocol."""

import os
from unittest.mock import patch

from rubrify.client import ChatClient, Client


class TestClientConstruction:
    def test_defaults(self) -> None:
        c = Client()
        assert c.base_url == ""
        assert c.api_key == ""

    def test_with_args(self) -> None:
        c = Client(base_url="https://api.example.com/", api_key="sk-test")
        assert c.base_url == "https://api.example.com"  # Trailing slash stripped
        assert c.api_key == "sk-test"

    def test_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {"RUBRIFY_BASE_URL": "https://api.test.com", "RUBRIFY_API_KEY": "sk-env"},
        ):
            c = Client.from_env()
            assert c.base_url == "https://api.test.com"
            assert c.api_key == "sk-env"

    def test_from_env_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            c = Client.from_env()
            assert c.base_url == ""
            assert c.api_key == ""

    def test_context_manager(self) -> None:
        with Client() as c:
            assert isinstance(c, Client)


class TestChatClientProtocol:
    def test_client_is_instance(self) -> None:
        c = Client()
        assert isinstance(c, ChatClient)

    def test_custom_client_conforms(self) -> None:
        class MyClient:
            def chat(
                self,
                *,
                messages: list[dict[str, str]],
                model: str,
                temperature: float = 0.0,
                max_tokens: int = 4096,
            ) -> str:
                return "response"

        mc = MyClient()
        assert isinstance(mc, ChatClient)

    def test_non_conforming_class(self) -> None:
        class BadClient:
            def do_stuff(self) -> str:
                return "nope"

        bc = BadClient()
        assert not isinstance(bc, ChatClient)
