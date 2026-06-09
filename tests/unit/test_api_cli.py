"""Tests for CLI models command and plugin API endpoints."""
from __future__ import annotations

import json

import pytest


class TestCliModelsCommand:
    """``chat4000 models`` CLI command — must be registered and handle
    the no-Hermes case gracefully."""

    def test_models_command_registered(self):
        from chat4000_hermes_plugin.cli import _build_chat4000_cli
        group = _build_chat4000_cli()
        assert "models" in group.commands

    def test_models_command_help(self):
        from click.testing import CliRunner
        from chat4000_hermes_plugin.cli import _build_chat4000_cli
        runner = CliRunner()
        group = _build_chat4000_cli()
        result = runner.invoke(group, ["models", "--help"])
        assert result.exit_code == 0
        assert "List available models" in result.output

    def test_models_command_empty(self):
        """Without Hermes config, returns 'no models' message."""
        from click.testing import CliRunner
        from chat4000_hermes_plugin.cli import _build_chat4000_cli
        runner = CliRunner()
        group = _build_chat4000_cli()
        result = runner.invoke(group, ["models"])
        assert result.exit_code == 0
        assert "No models configured" in result.output


class TestPluginApi:
    """Plugin API endpoints — handler functions work without Hermes."""

    @pytest.mark.asyncio
    async def test_list_accounts_no_hermes(self):
        from chat4000_hermes_plugin.api import handle_list_accounts_raw
        result = await handle_list_accounts_raw()
        assert result["ok"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_list_models_no_hermes(self):
        from chat4000_hermes_plugin.api import handle_list_models_raw
        result = await handle_list_models_raw()
        assert result["ok"] is True
        assert result["models"] == []
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_upload_attachment_empty_body(self):
        from chat4000_hermes_plugin.api import handle_upload_attachment_raw
        result = await handle_upload_attachment_raw(None)
        assert result["ok"] is False
        assert "Empty request body" in result["error"]

    @pytest.mark.asyncio
    async def test_upload_attachment_invalid_json(self):
        from chat4000_hermes_plugin.api import handle_upload_attachment_raw
        result = await handle_upload_attachment_raw(b"not json")
        assert result["ok"] is False
        assert "Invalid JSON" in result["error"]

    @pytest.mark.asyncio
    async def test_upload_attachment_missing_filename(self):
        from chat4000_hermes_plugin.api import handle_upload_attachment_raw
        body = json.dumps({"data_base64": "aGVsbG8="}).encode()
        result = await handle_upload_attachment_raw(body)
        assert result["ok"] is False
        assert "Missing filename" in result["error"]

    @pytest.mark.asyncio
    async def test_upload_attachment_missing_data(self):
        from chat4000_hermes_plugin.api import handle_upload_attachment_raw
        body = json.dumps({"filename": "test.txt"}).encode()
        result = await handle_upload_attachment_raw(body)
        assert result["ok"] is False
        assert "Missing data_base64" in result["error"]

    @pytest.mark.asyncio
    async def test_upload_attachment_invalid_base64(self):
        from chat4000_hermes_plugin.api import handle_upload_attachment_raw
        body = json.dumps({
            "data_base64": "not-valid-base64!!!",
            "filename": "test.txt",
        }).encode()
        result = await handle_upload_attachment_raw(body)
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_upload_attachment_success(self):
        import base64
        from chat4000_hermes_plugin.api import handle_upload_attachment_raw
        body = json.dumps({
            "data_base64": base64.b64encode(b"hello world").decode(),
            "filename": "test.txt",
            "mime_type": "text/plain",
            "text": "caption",
        }).encode()
        result = await handle_upload_attachment_raw(body)
        assert result["ok"] is True
        assert result["ref"].startswith("attach:")
        assert result["filename"] == "test.txt"
        assert result["mime_type"] == "text/plain"
        assert result["size"] == 11
        assert result["text"] == "caption"
        assert result["path"] is not None

    def test_register_plugin_api_noop(self):
        """register_plugin_api logs but doesn't crash with a bare context."""
        from chat4000_hermes_plugin.api import register_plugin_api
        # Should not raise with any object
        register_plugin_api(None)
        register_plugin_api(object())
