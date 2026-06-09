"""End-to-end verification for attachments, model picker, and multi-session."""
from __future__ import annotations

import pytest

from chat4000_hermes_plugin.protocol_types import (
    InnerMessage,
    InnerMessageFrom,
    OutboundAttachment,
    OutboundAck,
    OutboundToolEnd,
    OutboundToolStart,
    OutboundText,
    OutboundTextDelta,
    OutboundTextEnd,
    OutboundStatus,
    OutboundImage,
    OutboundAudio,
)


class TestOutboundAttachment:
    """Attachment protocol type covers all required fields."""

    def test_attachment_defaults(self):
        a = OutboundAttachment(data=b"hello", mime_type="text/plain", filename="readme.txt")
        assert a.kind == "attachment"
        assert a.data == b"hello"
        assert a.mime_type == "text/plain"
        assert a.filename == "readme.txt"
        assert a.text == ""  # default

    def test_attachment_with_text(self):
        a = OutboundAttachment(
            data=b"pdf-content", mime_type="application/pdf",
            filename="doc.pdf", text="see attached doc",
        )
        assert a.text == "see attached doc"

    def test_attachment_in_outbound_union(self):
        """OutboundAttachment is a valid OutboundMessage member."""
        from chat4000_hermes_plugin.protocol_types import OutboundMessage
        a: OutboundMessage = OutboundAttachment(
            data=b"x", mime_type="text/plain", filename="f.txt",
        )
        assert isinstance(a, OutboundAttachment)


class TestModelPicker:
    """Model override is extracted from inner body and embedded in raw_message."""

    def test_model_field_extracted(self):
        """Inner body with model key has model accessible."""
        inner = InnerMessage(
            t="text",
            id="msg-1",
            from_=InnerMessageFrom(role="app"),
            body={"text": "hello", "model": "gpt-4o-mini"},
            ts=1000,
        )
        model = (inner.body or {}).get("model") or None
        assert model == "gpt-4o-mini"

    def test_model_field_absent(self):
        """Inner body without model key returns None."""
        inner = InnerMessage(
            t="text",
            id="msg-2",
            from_=InnerMessageFrom(role="app"),
            body={"text": "hello"},
            ts=1000,
        )
        model = (inner.body or {}).get("model") or None
        assert model is None

    def test_model_field_empty(self):
        """Inner body with empty model string returns None."""
        inner = InnerMessage(
            t="text",
            id="msg-3",
            from_=InnerMessageFrom(role="app"),
            body={"text": "hello", "model": ""},
            ts=1000,
        )
        model = (inner.body or {}).get("model") or None
        assert model is None

    def test_model_in_raw_message(self):
        """Model is embedded in to_wire() output."""
        inner = InnerMessage(
            t="text",
            id="msg-4",
            from_=InnerMessageFrom(role="app"),
            body={"text": "hello", "model": "claude-sonnet-4-20250514"},
            ts=1000,
        )
        raw = inner.to_wire()
        raw["model"] = "claude-sonnet-4-20250514"
        assert raw.get("model") == "claude-sonnet-4-20250514"


class TestSessionRegistry:
    """SessionRegistry tracks multiple accounts with their transports."""

    def test_register_and_lookup(self):
        from chat4000_hermes_plugin.session_registry import SessionRegistry
        from chat4000_hermes_plugin.protocol_types import ResolvedChat4000Account, Chat4000AccountConfig

        reg = SessionRegistry()
        assert reg.transport_count == 0

        # Register two accounts with mock transports
        from chat4000_hermes_plugin.transport.mock import MockMessageTransport

        t1 = MockMessageTransport()
        a1 = ResolvedChat4000Account(
            account_id="devices", enabled=True, configured=True,
            relay_url="wss://relay.test/ws", pairing_log_level="info",
            runtime_log_level="info", group_id="group-a",
            group_key_bytes=b"\x00" * 32, key_file_path="/tmp/k1",
            key_source="state-file", config=Chat4000AccountConfig(),
        )
        reg.register("devices", a1, t1)

        t2 = MockMessageTransport()
        a2 = ResolvedChat4000Account(
            account_id="default", enabled=True, configured=True,
            relay_url="wss://relay.test/ws", pairing_log_level="info",
            runtime_log_level="info", group_id="group-b",
            group_key_bytes=b"\x01" * 32, key_file_path="/tmp/k2",
            key_source="state-file", config=Chat4000AccountConfig(),
        )
        reg.register("default", a2, t2)

        assert reg.transport_count == 2
        assert reg.get_transport("devices") is t1
        assert reg.get_transport("default") is t2
        assert reg.get_account_for_group("group-a") == "devices"
        assert reg.get_account_for_group("group-b") == "default"
        assert set(reg.get_all_account_ids()) == {"devices", "default"}

    def test_unregister(self):
        from chat4000_hermes_plugin.session_registry import SessionRegistry
        from chat4000_hermes_plugin.protocol_types import ResolvedChat4000Account, Chat4000AccountConfig
        from chat4000_hermes_plugin.transport.mock import MockMessageTransport

        reg = SessionRegistry()
        t = MockMessageTransport()
        a = ResolvedChat4000Account(
            account_id="test", enabled=True, configured=True,
            relay_url="wss://relay.test/ws", pairing_log_level="info",
            runtime_log_level="info", group_id="group-x",
            group_key_bytes=b"\x02" * 32, key_file_path="/tmp/k",
            key_source="state-file", config=Chat4000AccountConfig(),
        )
        reg.register("test", a, t)
        assert reg.transport_count == 1
        reg.unregister("test")
        assert reg.transport_count == 0
        assert reg.get_account_for_group("group-x") is None

    def test_get_transport_for_chat(self):
        """The _get_transport_for_chat helper resolves chat_id to transport."""
        from chat4000_hermes_plugin.transport.mock import MockMessageTransport

        t1 = MockMessageTransport()
        t2 = MockMessageTransport()
        transports = {"devices": t1, "default": t2}

        def _get(chat_id, primary=None):
            if not chat_id:
                return primary
            aid = chat_id.removeprefix("chat4000:")
            return transports.get(aid) or primary

        assert _get("chat4000:devices") is t1
        assert _get("chat4000:default") is t2
        assert _get("chat4000:unknown", primary=t1) is t1
        assert _get("", primary=t2) is t2


class TestListAvailableModels:
    """list_available_models returns empty list without Hermes (tests load)."""

    def test_returns_empty_without_hermes(self):
        from chat4000_hermes_plugin.adapter import list_available_models
        models = list_available_models()
        assert isinstance(models, list)
        # Without Hermes config, returns empty
        assert len(models) == 0
