"""Plugin HTTP API endpoints for Swift client integration.

Exposes three endpoints that the chat4000 Swift client queries through
the Hermes gateway's web API:

  GET  /plugins/chat4000/accounts   — list paired accounts
  GET  /plugins/chat4000/models     — available models from provider config
  POST /plugins/chat4000/attach     — stage attachment file, return ref

Each handler is a pure async function that returns a JSON-serializable
dict. Registration is attempted at plugin startup; if the Hermes version
doesn't support plugin API routes, the call is silently skipped (the
handlers remain importable for manual wiring).
"""

from __future__ import annotations

import base64
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─── Handlers ────────────────────────────────────────────────────────────────


async def handle_list_accounts_raw(body: bytes | None = None) -> dict[str, Any]:
    """GET /plugins/chat4000/accounts → list of configured accounts.

    Reads Hermes config and returns each account's id, group_id prefix,
    key source, and configured/enabled state.
    """
    try:
        from .adapter import list_available_models  # noqa: F401 — warm import
        from .accounts import list_chat4000_account_ids, resolve_chat4000_account
        from gateway.config import load_config
    except Exception:
        return {"ok": False, "error": "Hermes config not available"}

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
            "relay_url": acct.relay_url,
        })

    return {
        "ok": True,
        "accounts": accounts,
        "count": len(accounts),
    }


async def handle_list_models_raw(body: bytes | None = None) -> dict[str, Any]:
    """GET /plugins/chat4000/models → available models.

    Reads configured providers from Hermes config and returns flat list
    with name / provider / group per model.
    """
    try:
        from .adapter import list_available_models
    except Exception:
        return {"ok": False, "error": "Plugin module not available"}

    models = list_available_models()
    return {
        "ok": True,
        "models": models,
        "count": len(models),
    }


async def handle_upload_attachment_raw(body: bytes | None = None) -> dict[str, Any]:
    """POST /plugins/chat4000/attach — stage a file for sending.

    Accepts raw POST body as JSON or multipart form data. Expects fields:

    - ``data_base64`` (required) — base64-encoded file content
    - ``filename`` (required) — original filename
    - ``mime_type`` (optional, default: application/octet-stream)
    - ``text`` (optional) — caption text

    Returns an attachment ref the Swift client includes in a subsequent
    ``OutboundAttachment`` inner message:

    .. code-block:: json

        {"ok": true, "ref": "attach:uuid", "path": "/tmp/.../file.pdf"}
    """
    if not body:
        return {"ok": False, "error": "Empty request body"}

    import json

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return {"ok": False, "error": "Invalid JSON"}

    data_b64: str = (payload.get("data_base64") or "").strip()
    filename: str = (payload.get("filename") or "").strip()
    mime_type: str = (payload.get("mime_type") or "application/octet-stream").strip()
    text: str = (payload.get("text") or "").strip()

    if not data_b64:
        return {"ok": False, "error": "Missing data_base64"}
    if not filename:
        return {"ok": False, "error": "Missing filename"}

    try:
        raw_bytes = base64.b64decode(data_b64)
    except Exception:
        return {"ok": False, "error": "Invalid base64"}

    # Cache to ~/.hermes/cache/attachments/ same as the inbound handler
    try:
        from .key_store import resolve_hermes_state_dir
        cache_dir = resolve_hermes_state_dir() / "cache" / "attachments"
    except Exception:
        import tempfile
        cache_dir = Path(tempfile.gettempdir()) / "chat4000-attachments"

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Unique filename to avoid collisions
    stem, suffix = os.path.splitext(filename)
    if not suffix:
        suffix = ".bin"
    ref_id = uuid.uuid4().hex[:12]
    dest = cache_dir / f"{stem}-{ref_id}{suffix}"

    try:
        dest.write_bytes(raw_bytes)
    except OSError as exc:
        return {"ok": False, "error": f"Write failed: {exc}"}

    attachment_ref = f"attach:{ref_id}"

    return {
        "ok": True,
        "ref": attachment_ref,
        "path": str(dest),
        "filename": filename,
        "mime_type": mime_type,
        "size": len(raw_bytes),
        "text": text,
    }


# ─── Registration ────────────────────────────────────────────────────────────

_ROUTE_HANDLERS: list[tuple[str, str, str]] = [
    ("GET", "/plugins/chat4000/accounts", "chat4000_hermes_plugin.api.handle_list_accounts_raw"),
    ("GET", "/plugins/chat4000/models", "chat4000_hermes_plugin.api.handle_list_models_raw"),
    ("POST", "/plugins/chat4000/attach", "chat4000_hermes_plugin.api.handle_upload_attachment_raw"),
]


def register_plugin_api(ctx: Any) -> None:
    """Register chat4000 API endpoints with the Hermes gateway.

    Tries Hermes' plugin API route registration if available; logs a
    warning on versions that don't support it. The handlers remain
    importable for manual wiring (e.g. via hermes-webui's plugin API)."""
    if hasattr(ctx, "register_api_handler"):
        for method, path, handler_dotted in _ROUTE_HANDLERS:
            try:
                ctx.register_api_handler(path, method, handler_dotted)
                logger.info("chat4000: registered %s %s", method, path)
            except Exception as exc:
                logger.warning(
                    "chat4000: failed to register %s %s: %s", method, path, exc,
                )
        return

    if hasattr(ctx, "register_api_route"):
        for method, path, handler_dotted in _ROUTE_HANDLERS:
            try:
                ctx.register_api_route(path, method, handler_dotted)
                logger.info("chat4000: registered %s %s", method, path)
            except Exception as exc:
                logger.warning(
                    "chat4000: failed to register %s %s: %s", method, path, exc,
                )
        return

    logger.info(
        "chat4000: plugin API routes not registered (no ctx.register_api_handler / "
        "register_api_route); handlers available at chat4000_hermes_plugin.api",
    )
