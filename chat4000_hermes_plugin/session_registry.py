"""Multi-account session registry — manages concurrent relay transports.

Each active account (chat4000 group) gets its own relay transport that
connects independently to the chat4000 relay. Inbound messages from any
group are routed to the correct Hermes agent session via the account_id.

The registry maintains:
  - accounts: account_id → resolved account config
  - transports: account_id → RelayMessageTransport
  - group_to_account: group_id → account_id  (inbound routing)

Hermes maintains per-(chat_id) sessions automatically (via
BasePlatformAdapter.handle_message). By using distinct chat_id values
of the form ``chat4000:{account_id}``, each account gets its own Hermes
agent session.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from .transport import MessageTransport
from .protocol_types import ResolvedChat4000Account

logger = logging.getLogger(__name__)


class SessionRegistry:
    """Tracks all active chat4000 sessions with their transport and config.

    Thread-safe: all mutations are done from the gateway's asyncio loop,
    which is single-threaded. No locks needed.
    """

    def __init__(self) -> None:
        self._accounts: dict[str, ResolvedChat4000Account] = {}
        self._transports: dict[str, MessageTransport] = {}
        self._group_to_account: dict[str, str] = {}

    # ─── Registration ───────────────────────────────────────────────────

    def register(
        self,
        account_id: str,
        account: ResolvedChat4000Account,
        transport: MessageTransport,
    ) -> None:
        """Register a transport for an account. Replaces existing if called
        again (e.g. on re-pairing with new keys)."""
        old = self._transports.get(account_id)
        if old is not None and old is not transport:
            try:
                result = old.disconnect()
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception:
                logger.exception("session_registry: stale transport disconnect failed")

        self._accounts[account_id] = account
        self._transports[account_id] = transport
        if account.group_id:
            self._group_to_account[account.group_id] = account_id

    def unregister(self, account_id: str) -> None:
        """Remove an account and its transport from the registry."""
        account = self._accounts.pop(account_id, None)
        if account and account.group_id:
            self._group_to_account.pop(account.group_id, None)
        self._transports.pop(account_id, None)

    def clear(self) -> None:
        """Disconnect all transports and clear state."""
        for account_id in list(self._transports.keys()):
            self.unregister(account_id)
        self._accounts.clear()
        self._group_to_account.clear()

    # ─── Lookups ────────────────────────────────────────────────────────

    def get_transport(self, account_id: str) -> Optional[MessageTransport]:
        return self._transports.get(account_id)

    def get_account(self, account_id: str) -> Optional[ResolvedChat4000Account]:
        return self._accounts.get(account_id)

    def get_account_for_group(self, group_id: str) -> Optional[str]:
        return self._group_to_account.get(group_id)

    def get_all_account_ids(self) -> list[str]:
        return list(self._accounts.keys())

    def get_all_group_ids(self) -> list[str]:
        return list(self._group_to_account.keys())

    def is_connected(self, account_id: str) -> bool:
        return account_id in self._transports

    @property
    def transport_count(self) -> int:
        return len(self._transports)

    @property
    def account_count(self) -> int:
        return len(self._accounts)


# Module-level singleton so the adapter, CLI, and hooks share one registry.
_REGISTRY = SessionRegistry()


def get_registry() -> SessionRegistry:
    return _REGISTRY


def reset_registry_for_tests() -> None:
    _REGISTRY.clear()
