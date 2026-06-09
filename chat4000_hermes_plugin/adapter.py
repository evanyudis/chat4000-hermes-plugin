"""Chat4000 platform adapter for Hermes — the entry point Hermes calls
via plugin.yaml + the platform registry.

This is the analog of clawconnect-plugin/src/channel.ts but rewritten
against Hermes' BasePlatformAdapter contract instead of OpenClaw's
defineBundledChannelEntry. Same protocol, same crypto, same relay —
just a different host SDK.

Responsibilities:
  - Implement BasePlatformAdapter (connect/disconnect/send/get_chat_info)
  - On inbound message: decrypt → dispatch to Hermes' agent runner
  - On outbound (agent → user): forward via StreamDispatcher
  - On tool-call lifecycle: forward via ToolCallDispatcher (NEW)
  - Emit Flow B inner ack on app-origin text/image/audio per §6.6.5

Hermes integration points:
  - register(ctx): registered platform name "chat4000", emoji 🔐,
    label "chat4000"
  - hooks `on_tool_start` / `on_tool_output` / `on_tool_end` from the
    agent reply pipeline (provided by Hermes core, exposed via
    `replyOptions` in the reply pipeline construction)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Optional

from .accounts import resolve_chat4000_account, list_chat4000_account_ids, get_default_chat4000_account_id
from .protocol_types import (
    ConnectionFailed,
    InnerMessage,
    OutboundAck,
    OutboundAudio,
    OutboundAttachment,
    OutboundImage,
    OutboundInfoResponse,
    OutboundStatus,
    OutboundText,
)

logger = logging.getLogger(__name__)

from .session_registry import get_registry
# Lazy imports below — Hermes' BasePlatformAdapter lives in the host
# process. We avoid importing at module top so test/CI runs without the
# Hermes core present still pass.


class Chat4000Adapter:  # subclass of BasePlatformAdapter, lazily resolved
    """The actual class declaration uses BasePlatformAdapter as base.
    We monkey-patch the bases at register() time so this module is
    importable without Hermes present (e.g. unit tests for crypto).

    The Hermes SDK contract:
      connect() -> bool        — async, return True on success
      disconnect()             — async, clean shutdown
      send(chat_id, content, *, reply_to=None, metadata=None) -> SendResult
      send_typing(chat_id)
      send_image(chat_id, image_url, caption) -> SendResult
      get_chat_info(chat_id) -> dict
    """

    def __init__(self, config, **kwargs):
        # Resolve Hermes types lazily so this module imports without Hermes.
        from gateway.platforms.base import BasePlatformAdapter  # type: ignore[import-not-found]
        from gateway.config import Platform  # type: ignore[import-not-found]

        # Hand-wired super call (the type-system-level inheritance is set
        # at register() time via _make_adapter_class).
        BasePlatformAdapter.__init__(
            self, config=config, platform=Platform("chat4000")
        )

        extra = getattr(config, "extra", {}) or {}
        self._account_id = extra.get("accountId") or extra.get("account_id") or "default"
        self._config = config
        self._cfg = extra  # raw extras for resolve_chat4000_account
        self._transport: Optional[RelayMessageTransport] = None
        self._abort_signal = asyncio.Event()
        self._stream_dispatcher: Optional[StreamDispatcher] = None
        self._tool_dispatcher: Optional[ToolCallDispatcher] = None
        self._connected = False
        self._handlers_unsubscribe: list = []
        # Captured at connect-time for plugin_hooks to schedule async
        # frame emissions on the right asyncio loop.
        extra = getattr(config, "extra", {}) or {}
        self._account_id = extra.get("accountId") or extra.get("account_id") or "default"
        self._config = config
        self._cfg = extra  # raw extras for resolve_chat4000_account
        self._transports: dict[str, RelayMessageTransport] = {}
        self._transport: Optional[RelayMessageTransport] = None

    @property
    def name(self) -> str:
        return "chat4000"

    # ─── BasePlatformAdapter — lifecycle ─────────────────────────────────

    async def connect(self) -> bool:
        """Connect ALL configured accounts, each with its own relay transport.

        Each account gets an independent WebSocket connection to the relay
        and routes to a separate Hermes agent session (via distinct chat_id).
        """
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        # Resolve accounts from config merged with env.
        synthetic_cfg = {"channels": {"chat4000": {"accounts": {}}}}
        if self._cfg:
            synthetic_cfg["channels"]["chat4000"]["accounts"][self._account_id] = self._cfg

        account_ids = list_chat4000_account_ids(synthetic_cfg)
        if not account_ids:
            account_ids = [self._account_id]

        connected_any = False
        for aid in account_ids:
            account = resolve_chat4000_account(synthetic_cfg, aid)
            if not account.configured:
                logger.info(
                    "chat4000: account %r not configured — skipping. Run `hermes chat4000 pair`.",
                    aid,
                )
                continue

            transport = RelayMessageTransport(abort_signal=self._abort_signal)
            register_transport(aid, transport)
            self._transports[aid] = transport

            # Wrap receive handler to capture the account_id context
            def _wire_handlers(aid_: str, t: RelayMessageTransport) -> None:
                nonlocal connected_any  # type: ignore[assignment]
                unsub_recv = t.on_receive(
                    lambda inner: self._on_inner_received(inner, account_id=aid_, transport=t)
                )
                unsub_state = t.on_connection_state(
                    lambda s: self._on_connection_state(aid_, s)
                )
                self._handlers_unsubscribe.extend([unsub_recv, unsub_state])

            _wire_handlers(aid, transport)

            transport.connect(
                TransportGroupConfig(
                    account_id=account.account_id,
                    group_id=account.group_id,
                    group_key_bytes=account.group_key_bytes,
                    relay_url=account.relay_url,
                    release_channel=account.config.release_channel,
                    runtime_log_level=account.runtime_log_level,
                )
            )
            get_registry().register(aid, account, transport)
            connected_any = True

        if not connected_any:
            logger.error(
                "chat4000: no configured accounts found — run `hermes chat4000 pair`"
            )
            return False

        # Primary transport for tool dispatcher and backward compat
        primary = self._transports.get(self._account_id)
        if primary is None:
            primary = next(iter(self._transports.values()), None)
        self._transport = primary
        if primary is not None:
            self._tool_dispatcher = ToolCallDispatcher(
                send=lambda msg: primary.send(msg) if primary else None,  # type: ignore[union-attr]
            )

        self._mark_connected()
        self._connected = True
        from . import analytics
        analytics.track("gateway_started", {"account_count": len(self._transports)})
        return True

    async def disconnect(self) -> None:
        """Disconnect ALL transports and reset state."""
        self._connected = False
        from . import analytics
        analytics.track("gateway_stopped", {})
        analytics.flush()
        from .plugin_hooks import deregister_active_adapter
        deregister_active_adapter(self)
        self._abort_signal.set()
        for unsub in self._handlers_unsubscribe:
            try:
                unsub()
            except Exception:
                pass
        self._handlers_unsubscribe.clear()
        if self._stream_dispatcher is not None:
            self._stream_dispatcher.dispose()
            self._stream_dispatcher = None
        if self._tool_dispatcher is not None:
            self._tool_dispatcher.dispose()
            self._tool_dispatcher = None
        # Disconnect all transports
        for aid, transport in list(self._transports.items()):
            try:
                await transport.disconnect()
            except Exception:
                pass
            unregister_transport(aid)
            get_registry().unregister(aid)
        self._transports.clear()
        self._transport = None
        try:
            self._mark_disconnected()
        except Exception:
            pass
    async def send(
        self,
        chat_id: str,
        content,
        *,
        reply_to: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        """Hermes calls this when the agent has a final reply to deliver.

        Uses ``chat_id`` to route to the correct account's transport.
        Falls back to ``self._account_id`` when chat_id doesn't match any
        known transport (backward compat)."""
        from gateway.platforms.base import SendResult  # type: ignore[import-not-found]

        transport = self._get_transport_for_chat(chat_id)
        if transport is None:
            return SendResult(success=False, error="transport not connected")

        if isinstance(content, dict):
            text = content.get("text", "") or ""
            media_url = content.get("media_url")
        else:
            text = str(content or "")
            media_url = None

        if media_url:
            text = (text + ("\n\n" if text else "") + f"Attachment: {media_url}").strip()

        if not text:
            return SendResult(success=True, message_id="")

        wire_id = transport.send(OutboundText(text=text))
        return SendResult(success=True, message_id=wire_id)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        transport = self._get_transport_for_chat(chat_id)
        if transport is None:
            return
        transport.send(OutboundStatus(status="typing"))

    async def send_image(self, chat_id, image_url, caption=None):
        from gateway.platforms.base import SendResult  # type: ignore[import-not-found]
        text = (caption + "\n\n" if caption else "") + f"Image: {image_url}"
        transport = self._get_transport_for_chat(chat_id)
        if transport is None:
            return SendResult(success=False, error="transport not connected")
        wire_id = transport.send(OutboundText(text=text))
        return SendResult(success=True, message_id=wire_id)

    def _get_transport_for_chat(self, chat_id: str) -> Optional[RelayMessageTransport]:
        """Return the relay transport for a given chat_id.

        The chat_id format is ``chat4000:{account_id}``. Falls back to
        ``self._account_id`` stripped of prefix, then to the primary transport.
        """
        if not chat_id:
            return self._transport
        # Strip "chat4000:" prefix if present
        aid = chat_id.removeprefix("chat4000:")
        return self._transports.get(aid) or self._transport

    async def get_chat_info(self, chat_id) -> dict:
        return {"name": f"chat4000 ({chat_id[:8]}...)", "type": "dm", "chat_id": chat_id}

    # ─── Hermes reply-pipeline hooks (streaming + tool calls) ────────────

    def reply_pipeline_options(self) -> dict:
        """Returned to Hermes' reply-pipeline factory so the agent's
        per-turn streaming + tool-call events flow into our dispatchers.

        Hermes' reply pipeline currently exposes:
          - on_reasoning_stream / on_reasoning_end
          - on_assistant_message_start
          - on_partial_reply(payload: { text })
          - on_tool_start(name, args)
          - on_tool_output(tool_id, delta)         [may not always fire]
          - on_tool_end(tool_id, status, result)

        We map these into chat4000 wire frames."""
        if self._transport is None:
            return {}

        # Fresh dispatchers per turn.
        self._stream_dispatcher = StreamDispatcher(
            send=lambda msg: self._transport.send(msg) if self._transport else None,  # type: ignore[union-attr]
        )
        self._tool_dispatcher = ToolCallDispatcher(
            send=lambda msg: self._transport.send(msg) if self._transport else None,  # type: ignore[union-attr]
        )

        # Hermes' agent pipeline doesn't natively expose a per-call tool_id
        # the way our dispatcher wants — we mint one on tool_start and
        # thread it through. Maintain a per-turn name→tool_id map for the
        # cases where on_tool_output/on_tool_end only carry the tool name.
        active_tool_ids_by_name: dict[str, list[str]] = {}

        async def on_reasoning_stream(_payload: dict) -> None:
            self._transport.send(OutboundStatus(status="thinking"))  # type: ignore[union-attr]

        async def on_reasoning_end(_payload: dict) -> None:
            self._transport.send(OutboundStatus(status="typing"))  # type: ignore[union-attr]

        async def on_assistant_message_start(_payload: dict) -> None:
            self._transport.send(OutboundStatus(status="typing"))  # type: ignore[union-attr]

        async def on_partial_reply(payload: dict) -> None:
            text = payload.get("text") or ""
            if not text:
                return
            self._transport.send(OutboundStatus(status="typing"))  # type: ignore[union-attr]
            if self._stream_dispatcher is not None:
                await self._stream_dispatcher.on_partial(text)

        async def on_tool_start(name: str, args) -> str:
            if self._tool_dispatcher is None:
                return ""
            self._transport.send(OutboundStatus(status="thinking"))  # type: ignore[union-attr]
            tool_id = await self._tool_dispatcher.on_tool_start(name=name, args=args)
            active_tool_ids_by_name.setdefault(name, []).append(tool_id)
            return tool_id

        async def on_tool_output(tool_id_or_name: str, delta: str) -> None:
            if self._tool_dispatcher is None:
                return
            tool_id = _resolve_tool_id(tool_id_or_name, active_tool_ids_by_name)
            if tool_id is None:
                return
            await self._tool_dispatcher.on_tool_output(tool_id, delta)

        async def on_tool_end(tool_id_or_name: str, *, status: str = "done", result: str = "") -> None:
            if self._tool_dispatcher is None:
                return
            tool_id = _resolve_tool_id(tool_id_or_name, active_tool_ids_by_name, pop=True)
            if tool_id is None:
                return
            await self._tool_dispatcher.on_tool_end(
                tool_id, status=status, result=result  # type: ignore[arg-type]
            )

        async def on_final(payload: dict) -> None:
            text = payload.get("text") or ""
            if self._stream_dispatcher is None:
                return
            outcome = await self._stream_dispatcher.on_final(text)
            if outcome == "oneshot" and text:
                self._transport.send(OutboundText(text=text))  # type: ignore[union-attr]
            self._transport.send(OutboundStatus(status="idle"))  # type: ignore[union-attr]

        return {
            "on_reasoning_stream": on_reasoning_stream,
            "on_reasoning_end": on_reasoning_end,
            "on_assistant_message_start": on_assistant_message_start,
            "on_partial_reply": on_partial_reply,
            "on_tool_start": on_tool_start,
            "on_tool_output": on_tool_output,
            "on_tool_end": on_tool_end,
            "on_final": on_final,
        }

    # ─── Inbound dispatch ────────────────────────────────────────────────

    def _on_inner_received(
        self,
        inner: InnerMessage,
        account_id: str = "",
        transport: Optional[MessageTransport] = None,
    ) -> Any:
        """Called once per decrypted+dedup'd inbound inner message.

        ``account_id`` identifies which chat4000 group this message arrived
        on, so we route to the correct Hermes agent session."""
        is_from_app = inner.from_ is not None and inner.from_.role == "app"

        if inner.t == "ack":
            # Plugin-side acks not used in v1.
            return

        if inner.t in ("text_delta", "text_end", "status", "tool_start", "tool_delta", "tool_end"):
            # Anything we don't dispatch into the agent runner.
            return

        if inner.t == "info_request":
            # Handle info requests directly — no Hermes agent dispatch.
            # Client queries for accounts, models, and status go through the
            # same encrypted relay rather than a separate HTTP API.
            return asyncio.ensure_future(
                self._handle_info_request(inner, transport=transport)
            )

        if inner.t not in ("text", "image", "audio", "attachment"):
            return

        # Emit Flow B inner ack BEFORE running the agent so the iPhone
        # ✓✓ tick lights up immediately, not after token generation.
        tr = transport or self._transport
        if is_from_app and tr is not None:
            try:
                tr.send(OutboundAck(refs=inner.id, stage="received"))
            except Exception:
                pass

        # Dispatch to the Hermes agent runner via BasePlatformAdapter.
        return asyncio.ensure_future(
            self._dispatch_to_agent(inner, account_id=account_id)
        )

    async def _handle_info_request(
        self,
        inner: InnerMessage,
        transport: Optional[MessageTransport] = None,
    ) -> None:
        """Respond to an info_request from the Swift client over the relay.

        Dispatch is by ``body.type``. Supported types:

        - ``accounts`` — list of configured accounts with state
        - ``models`` — available models from Hermes provider config
        - ``status`` — adapter status (connected accounts, relay health)
        """
        req_type = (inner.body or {}).get("type", "")
        tr = transport or self._transport

        if req_type == "accounts":
            try:
                from .accounts import resolve_chat4000_account  # noqa: F811
                from gateway.config import load_config
                cfg = load_config()
                ids = list_chat4000_account_ids(cfg)
                accounts = []
                for aid in ids:
                    acct = resolve_chat4000_account(cfg, aid)
                    accounts.append({
                        "account_id": acct.account_id,
                        "enabled": acct.enabled,
                        "configured": acct.configured,
                        "group_id": acct.group_id or "",
                        "key_source": acct.key_source,
                    })
                if tr is not None:
                    tr.send(OutboundInfoResponse(
                        body={"type": "accounts", "accounts": accounts},
                        ref=inner.id,
                    ))
            except Exception as exc:
                logger.warning("chat4000: info_request accounts failed: %s", exc)

        elif req_type == "models":
            try:
                models = list_available_models()
                if tr is not None:
                    tr.send(OutboundInfoResponse(
                        body={"type": "models", "models": models},
                        ref=inner.id,
                    ))
            except Exception as exc:
                logger.warning("chat4000: info_request models failed: %s", exc)

        elif req_type == "status":
            try:
                connected_accounts = list(self._transports.keys()) if hasattr(self, "_transports") else []
                if tr is not None:
                    tr.send(OutboundInfoResponse(
                        body={"type": "status", "connected_accounts": connected_accounts},
                        ref=inner.id,
                    ))
            except Exception as exc:
                logger.warning("chat4000: info_request status failed: %s", exc)

    async def _dispatch_to_agent(
        self, inner: InnerMessage, account_id: str = ""
    ) -> None:
        """Hand the inbound text/image/audio to Hermes.

        ``account_id`` identifies which chat4000 group this message was
        received on; used to route to the correct Hermes agent session
        via a distinct chat_id in the SessionSource.
        it constructs a MessageEvent, builds the SessionSource, and routes
        through the gateway's session-resolution + agent-dispatch pipeline.
        That's the exact path the Telegram/Slack/Discord adapters take.

        For image/audio: Hermes' vision + STT tools read the file from
        `event.media_urls`, NOT from any payload dict. We cache the
        decoded bytes to `~/.hermes/cache/{images,audio}/` and pass the
        absolute path through — identical to how Telegram, WhatsApp,
        Discord etc. surface user attachments."""
        import base64

        from gateway.platforms.base import (  # type: ignore[import-not-found]
            MessageEvent,
            MessageType,
            cache_audio_from_bytes,
            cache_image_from_bytes,
        )

        media_urls: list[str] = []
        media_types: list[str] = []
        text = ""

        # Map inner.body → MessageEvent. For media types we decode the
        # base64 payload and cache it; Hermes' STT / vision tools pick
        # up the path from media_urls.
        if inner.t == "text":
            text = (inner.body or {}).get("text", "")
            message_type = MessageType.TEXT
        elif inner.t == "image":
            message_type = MessageType.IMAGE
            mime = (inner.body or {}).get("mime_type", "image/jpeg")
            data_b64 = (inner.body or {}).get("data_base64", "")
            if data_b64:
                try:
                    raw = base64.b64decode(data_b64)
                    ext = "." + (mime.rsplit("/", 1)[-1] or "jpg").lower()
                    # Common ones: image/jpeg → .jpg; image/png → .png.
                    if ext == ".jpeg":
                        ext = ".jpg"
                    media_urls.append(cache_image_from_bytes(raw, ext=ext))
                    media_types.append(mime)
                except Exception as exc:
                    logger.warning("chat4000: failed to cache inbound image: %s", exc)
        elif inner.t == "audio":
            message_type = MessageType.AUDIO
            mime = (inner.body or {}).get("mime_type", "audio/m4a")
            data_b64 = (inner.body or {}).get("data_base64", "")
            if data_b64:
                try:
                    raw = base64.b64decode(data_b64)
                    # iOS app sends `audio/m4a` for AVAudioRecorder output;
                    # macOS app + CLI may send `audio/ogg` or `audio/mp3`.
                    # Use the mime subtype as the extension so Hermes' STT
                    # picks the right decoder.
                    subtype = (mime.rsplit("/", 1)[-1] or "m4a").lower()
                    ext = "." + (subtype.split(";")[0].strip() or "m4a")
                    media_urls.append(cache_audio_from_bytes(raw, ext=ext))
                    media_types.append(mime)
                    logger.info(
                        "chat4000: cached inbound audio (%s, %d bytes) → %s",
                        mime, len(raw), media_urls[-1],
                    )
                except Exception as exc:
                    logger.warning("chat4000: failed to cache inbound audio: %s", exc)
        elif inner.t == "attachment":
            message_type = MessageType.TEXT  # Treat as text with attached file
            mime = (inner.body or {}).get("mime_type", "application/octet-stream")
            filename = (inner.body or {}).get("filename", "attachment")
            text = (inner.body or {}).get("text", "")
            data_b64 = (inner.body or {}).get("data_base64", "")
            if data_b64:
                try:
                    raw = base64.b64decode(data_b64)
                    # Cache to ~/.hermes/cache/attachments/ with original filename
                    from ..key_store import resolve_hermes_state_dir
                    cache_dir = resolve_hermes_state_dir() / "cache" / "attachments"
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    import shutil
                    dest = cache_dir / filename
                    # Avoid overwriting existing files — append counter if needed
                    if dest.exists():
                        stem = dest.stem
                        suffix = dest.suffix
                        counter = 1
                        while (cache_dir / f"{stem}_{counter}{suffix}").exists():
                            counter += 1
                        dest = cache_dir / f"{stem}_{counter}{suffix}"
                    dest.write_bytes(raw)
                    media_urls.append(str(dest))
                    media_types.append(mime)
                    logger.info(
                        "chat4000: cached inbound attachment (%s, %s, %d bytes) → %s",
                        filename, mime, len(raw), dest,
                    )
                except Exception as exc:
                    logger.warning("chat4000: failed to cache inbound attachment: %s", exc)
        else:
            return

        # Extract optional model override from the app-sent inner body.
        # When present, embed it in raw_message so Hermes' gateway can
        # route to the requested model for this turn.
        model = (inner.body or {}).get("model") or None
        if model:
            logger.info(
                "chat4000: model override: %s (inner.id=%s)",
                model, inner.id,
            )

        # Build the SessionSource via BasePlatformAdapter helper so the
        # gateway recognises us as a regular platform.
        # Use ``chat4000:{account_id}`` as the chat_id so each account
        # gets its own Hermes agent session.
        effective_account = account_id or self._account_id
        source = self.build_source(
            chat_id=f"chat4000:{effective_account}",
            user_id=(inner.from_.device_id if inner.from_ else None) or effective_account,
            chat_type="dm",
        )

        raw_msg = inner.to_wire()
        if model:
            raw_msg["model"] = model
        event = MessageEvent(
            text=text,
            message_type=message_type,
            source=source,
            raw_message=raw_msg,
            message_id=inner.id,
            media_urls=media_urls,
            media_types=media_types,
        )
        # handle_message is BasePlatformAdapter's bridge into the gateway
        # runner. The runner then constructs the agent, sets up the reply
        # pipeline with our `reply_pipeline_options`, and ships a final
        # `deliver` back into self.send(...).
        await self.handle_message(event)

    def _on_connection_state(self, state: Any) -> None:
        if isinstance(state, dict) and state.get("kind") == "failed":
            logger.warning("chat4000 relay failed: %s", state.get("reason"))
        elif state == "connected":
            logger.info("chat4000 connected to relay")
        elif state in ("disconnected", "reconnecting"):
            logger.info("chat4000 relay state: %s", state)


# ─── Hermes entry points ──────────────────────────────────────────────────


def _resolve_tool_id(
    candidate: str,
    map_by_name: dict[str, list[str]],
    *,
    pop: bool = False,
) -> Optional[str]:
    """Hermes' tool callbacks may pass either the tool_id we returned from
    on_tool_start, OR the tool name (depending on which hook point fired).
    Try the candidate as an id first; fall back to the FIFO of active
    tool_ids for that name."""
    if any(candidate == tid for tids in map_by_name.values() for tid in tids):
        # It IS a tool_id; pop from whichever name's list owns it.
        if pop:
            for tids in map_by_name.values():
                if candidate in tids:
                    tids.remove(candidate)
                    break
        return candidate
    tids = map_by_name.get(candidate)
    if not tids:
        return None
    return tids.pop(0) if pop else tids[0]


def check_requirements() -> bool:
    """Hermes calls this at adapter discovery to gate loading. The plugin
    is always loadable — the user just won't get a working group until
    they run `hermes chat4000 pair`."""
    return True


def validate_config(config) -> bool:
    """Whether the platform is configured well enough to connect. We accept
    either an env var override or a key file on disk. Both are checked by
    resolve_chat4000_account()."""
    # CHAT4000_GROUP_KEY env or a stored key file under ~/.hermes/plugins/chat4000/
    if os.getenv("CHAT4000_GROUP_KEY", "").strip():
        return True
    from .accounts import resolve_chat4000_account

    account = resolve_chat4000_account(None, None)
    return account.configured


def _env_enablement() -> Optional[dict]:
    """Auto-enable the platform when CHAT4000_GROUP_KEY is set or a key
    file exists. Hermes calls this BEFORE adapter construction so
    `hermes gateway status` sees the right state.

    Side-effect: also seeds `CHAT4000_HOME_CHANNEL` in the process env
    so Hermes' "📬 No home channel set" first-message prompt doesn't
    fire. chat4000 has exactly one group per gateway — the group_id
    IS the home channel, no user choice to make. Pre-existing
    CHAT4000_HOME_CHANNEL values are respected (operator override)."""
    from .accounts import resolve_chat4000_account

    account = resolve_chat4000_account(None, None)
    if not account.configured:
        return None
    if not os.getenv("CHAT4000_HOME_CHANNEL", "").strip():
        os.environ["CHAT4000_HOME_CHANNEL"] = account.group_id
    return {
        "accountId": account.account_id,
        "groupId": account.group_id,
        "home_channel": {"chat_id": account.group_id, "name": "chat4000"},
    }


def _make_adapter_class():
    """Hermes' BasePlatformAdapter is only importable from inside the
    Hermes process. Build the real class dynamically so the module
    imports cleanly during unit tests / CI."""
    from gateway.platforms.base import BasePlatformAdapter  # type: ignore[import-not-found]

    # Preserve everything except a few class-machinery dunders Python
    # populates automatically. In particular, keep `__init__` — without
    # it the dynamic class inherits BasePlatformAdapter.__init__, which
    # needs a `platform` positional arg the factory doesn't pass.
    _SKIP = {"__dict__", "__weakref__", "__module__", "__qualname__"}
    namespace = {
        k: v for k, v in Chat4000Adapter.__dict__.items() if k not in _SKIP
    }
    return type(
        "Chat4000Adapter",
        (BasePlatformAdapter,),
        namespace,
    )


def register(ctx) -> None:
    """Plugin entry point — Hermes' plugin loader calls this once on
    discovery. We register our platform via ctx.register_platform.

    The registry call also wires CLI subcommands (`hermes chat4000 ...`)
    — those live in src/cli.py and use the same ctx.register_cli surface
    Hermes' built-in plugins use."""
    from . import analytics
    from .plugin_hooks import register_plugin_hooks
    from .telemetry import initialize_chat4000_telemetry

    initialize_chat4000_telemetry()
    analytics.initialize_chat4000_analytics()
    analytics.set_person_properties({
        "plugin_version": analytics.PACKAGE_VERSION,
        "os_platform": __import__("sys").platform,
    })

    # Wire Hermes' cross-cutting tool-call hooks so the iOS app sees
    # tool_start / tool_end bubbles for every tool the agent invokes
    # in chat4000 sessions. The hooks self-filter by session_id.
    register_plugin_hooks(ctx)

    # chat4000's auth IS the 32-byte group key: anyone with it can already
    # decrypt every message. Hermes' per-user pairing on top would just
    # mean "pair the device once with chat4000 E2E, then ALSO ask the bot
    # owner to approve a code". Skip the second layer by default — users
    # who want platform-level allowlists can set CHAT4000_ALLOW_ALL_USERS
    # to "false" and use CHAT4000_ALLOWED_USERS like other platforms.
    if "CHAT4000_ALLOW_ALL_USERS" not in os.environ:
        os.environ["CHAT4000_ALLOW_ALL_USERS"] = "true"

    AdapterClass = _make_adapter_class()
    # Chat4000Adapter.__init__ does the BasePlatformAdapter super-call
    # itself (with `platform=Platform("chat4000")`) — factory only passes
    # the PlatformConfig.
    ctx.register_platform(
        name="chat4000",
        label="chat4000",
        adapter_factory=lambda cfg: AdapterClass(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        env_enablement_fn=_env_enablement,
        required_env=[],
        install_hint="Run `chat4000 pair` to pair a device.",
        max_message_length=4096,
        # Wire Hermes' auth allowlist envs to chat4000-specific names so
        # the gateway's _is_user_authorized branch resolves them via the
        # plugin registry instead of falling through to GATEWAY_*.
        allowed_users_env="CHAT4000_ALLOWED_USERS",
        allow_all_env="CHAT4000_ALLOW_ALL_USERS",
        platform_hint=(
            "You are chatting via chat4000 (encrypted iOS/macOS/CLI client). "
            "It supports markdown formatting and streams replies as text_delta "
            "frames. Tool calls render natively in the chat as expandable "
            "bubbles — keep tool args readable."
        ),
        emoji="🔐",
    )

    # CLI subcommands (hermes chat4000 pair / setup / ...) live in .cli
    # and are gated behind ctx.register_cli, which isn't present on
    # every Hermes version. Skip registration when the surface is
    # missing rather than crashing the whole plugin load.
    if hasattr(ctx, "register_cli"):
        from .cli import register_chat4000_cli
        register_chat4000_cli(ctx)

    # API endpoints for Swift client integration. The Hermes gateway may
    # or may not support plugin API route registration depending on the
    # version — log and continue either way.
    try:
        from .api import register_plugin_api
        register_plugin_api(ctx)
    except Exception:
        logger.info("chat4000: plugin API routes skipped (not supported by this Hermes version)")

def list_available_models() -> list[dict]:
    """Return available models from Hermes' provider config.

    Reads the configured providers and their models from Hermes' config
    system. Returns a list of dicts with ``name``, ``provider``, and
    ``group`` keys the Swift client can surface for model selection.

    Returns an empty list if Hermes config is not readable (e.g. during
    unit tests or cold install).
    """
    try:
        from gateway.config import load_config  # type: ignore[import-not-found]
        config = load_config()
        models = []
        providers = (config or {}).get("providers") or {}
        for provider_name, provider_cfg in providers.items():
            raw_models = provider_cfg.get("models") or provider_cfg.get("model") or []
            if isinstance(raw_models, str):
                raw_models = [raw_models]
            for model_name in raw_models:
                if isinstance(model_name, str) and model_name.strip():
                    models.append({
                        "name": model_name.strip(),
                        "provider": provider_name,
                        "group": provider_cfg.get("group", "default"),
                    })
        return models
    except Exception:
        return []