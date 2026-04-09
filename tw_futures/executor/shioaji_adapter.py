"""Shioaji (永豐金) adapter for TAIFEX futures trading.

Wraps the shioaji SDK (v1.3+) for Taiwan index futures (TXF 大台 / MXF 小台).

Simulation vs Live
------------------
* ``simulation=True``  — logs in with API Key + Secret only; no certificate
  required.  Orders are executed in the paper-trading environment.
* ``simulation=False`` — live trading; ``cert_path``, ``cert_password``, and
  ``person_id`` are required.  ``activate_ca()`` is called automatically after
  login.

Retry logic
-----------
All broker calls are wrapped with :func:`_with_retry`, which retries up to
``_MAX_RETRIES`` times with exponential back-off (2 s, 4 s, 8 s).

Error handling
--------------
All public methods raise :exc:`ExecutionError` on unrecoverable failures.
"""

from __future__ import annotations

import logging
import time
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0  # seconds; doubles each attempt


class ExecutionError(RuntimeError):
    """Raised when a broker API call fails after all retries."""


class OrderResult(NamedTuple):
    """Normalised result returned by ShioajiAdapter.submit_order."""

    order_id: str
    status: str
    action: str
    contracts: int
    price_type: str
    price: float


# ---------------------------------------------------------------------------
# Internal retry helper
# ---------------------------------------------------------------------------


def _with_retry(fn, *args, max_retries: int = _MAX_RETRIES, **kwargs) -> Any:
    """Call *fn* with retries on exception.

    Raises the last exception if all attempts fail.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = _BACKOFF_BASE**attempt
                logger.warning(
                    "ShioajiAdapter: attempt %d/%d failed (%s). Retrying in %.0fs …",
                    attempt,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "ShioajiAdapter: all %d attempts exhausted. Last error: %s",
                    max_retries,
                    exc,
                )
    raise ExecutionError(f"Broker call failed after {max_retries} retries") from last_exc


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class ShioajiAdapter:
    """Shioaji broker adapter for TAIFEX futures.

    Parameters
    ----------
    api_key :
        Shioaji API key (永豐 API 金鑰).
    secret_key :
        Shioaji secret key.
    cert_path :
        Path to the ``*.pfx`` certificate file.  Required for live trading
        (``simulation=False``); ignored in simulation mode.
    cert_password :
        Password for the certificate file.  Required for live trading.
    person_id :
        National ID / passport number tied to the certificate.  Required for
        live trading.
    simulation :
        If ``True`` (default), connects to the paper-trading environment.
        No certificate is needed.
    """

    #: Number of shares per lot (used for stock-mode; futures use contract count)
    LOT_SIZE: int = 1000

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        cert_path: str | None = None,
        cert_password: str | None = None,
        person_id: str | None = None,
        simulation: bool = True,
    ) -> None:
        if not simulation:
            missing = []
            if not cert_path:
                missing.append("cert_path")
            if not cert_password:
                missing.append("cert_password")
            if not person_id:
                missing.append("person_id")
            if missing:
                raise ValueError(f"Live trading (simulation=False) requires: {', '.join(missing)}")

        self.api_key = api_key
        self.secret_key = secret_key
        self.cert_path = cert_path
        self.cert_password = cert_password
        self.person_id = person_id
        self.simulation = simulation

        self._api = None  # shioaji.Shioaji, set in _connect()
        self._accounts = []  # list[Account], set after login

        self._connect()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Create API instance, login, and (for live) activate certificate.

        Login and contract download are separated so that a transient contract
        fetch failure does not cause the login step to be retried (login is
        not idempotent).
        """
        import shioaji as sj

        logger.info("ShioajiAdapter: connecting (simulation=%s) …", self.simulation)
        try:
            api = sj.Shioaji(simulation=self.simulation)
            # login without contract download first (idempotent-safe)
            accounts = api.login(
                self.api_key,
                self.secret_key,
                fetch_contract=False,
            )
        except Exception as exc:
            raise ExecutionError(f"Login failed: {exc}") from exc

        # Fetch contracts separately (retriable, contracts_timeout=0 = no limit)
        try:
            _with_retry(api.fetch_contracts, contracts_timeout=0)
        except ExecutionError as exc:
            logger.warning(
                "ShioajiAdapter: contract download failed (%s). "
                "Contract lookup will be unavailable.",
                exc,
            )

        if not self.simulation and self.cert_path:
            try:
                ok = api.activate_ca(
                    ca_path=self.cert_path,
                    ca_passwd=self.cert_password,
                    person_id=self.person_id or "",
                )
                if not ok:
                    raise ExecutionError("activate_ca() returned False")
                logger.info("ShioajiAdapter: CA certificate activated.")
            except ExecutionError:
                raise
            except Exception as exc:
                raise ExecutionError(f"Certificate activation failed: {exc}") from exc

        self._api = api
        self._accounts = accounts if accounts else []
        logger.info("ShioajiAdapter: logged in. accounts=%d", len(self._accounts))

    def logout(self) -> None:
        """Logout and release broker connection."""
        if self._api is not None:
            try:
                self._api.logout()
                logger.info("ShioajiAdapter: logged out.")
            except Exception as exc:
                logger.warning("ShioajiAdapter: logout warning: %s", exc)
            finally:
                self._api = None

    def __del__(self) -> None:
        self.logout()

    # ------------------------------------------------------------------
    # Account information
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        """Return futures account summary.

        Returns
        -------
        dict
            Keys: ``equity``, ``margin_used`` (initial margin),
            ``available_margin``, ``unrealized_pnl``.
        """
        try:
            fut_acct = self._futures_account()
            margin = _with_retry(self._api.margin, account=fut_acct)
        except ExecutionError:
            raise
        except Exception as exc:
            raise ExecutionError(f"get_account failed: {exc}") from exc

        return {
            "equity": margin.equity,
            "margin_used": margin.initial_margin,
            "available_margin": margin.available_margin,
            "unrealized_pnl": margin.future_open_position,
        }

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> list[dict]:
        """Return all open futures positions.

        Returns
        -------
        list[dict]
            Each dict has keys: ``code``, ``direction``, ``contracts``,
            ``avg_price``, ``last_price``, ``unrealized_pnl``.
        """
        try:
            fut_acct = self._futures_account()
            positions = _with_retry(self._api.list_positions, account=fut_acct)
        except ExecutionError:
            raise
        except Exception as exc:
            raise ExecutionError(f"get_positions failed: {exc}") from exc

        result = []
        for pos in positions:
            result.append(
                {
                    "code": pos.code,
                    "direction": pos.direction.value,  # "Buy" | "Sell"
                    "contracts": pos.quantity,
                    "avg_price": pos.price,
                    "last_price": pos.last_price,
                    "unrealized_pnl": pos.pnl,
                }
            )
        return result

    # ------------------------------------------------------------------
    # Contract lookup
    # ------------------------------------------------------------------

    def get_contract(self, product: str = "TXF"):
        """Return the near-month contract object for *product*.

        Parameters
        ----------
        product :
            ``"TXF"`` (大台指期) or ``"MXF"`` (小台指期).

        Returns
        -------
        shioaji.contracts.Future
            Near-month futures contract (lowest delivery_date that is still
            valid today).
        """
        try:
            contracts = getattr(self._api.Contracts.Futures, product, None)
        except Exception as exc:
            raise ExecutionError(f"Cannot access Contracts.Futures.{product}: {exc}") from exc

        if contracts is None:
            raise ExecutionError(
                f"Product {product!r} not found in Contracts.Futures. "
                f"Available: {dir(self._api.Contracts.Futures)}"
            )

        # contracts is iterable; pick the nearest delivery month
        import datetime as _dt

        today_str = _dt.date.today().strftime("%Y/%m/%d")

        valid = [c for c in contracts if c.delivery_date >= today_str]
        if not valid:
            # Fallback: return the last available (most recent) contract
            valid = list(contracts)

        # Sort ascending by delivery_date and pick the nearest
        valid.sort(key=lambda c: c.delivery_date)
        near_month = valid[0]
        logger.debug(
            "get_contract(%s): near-month = %s  delivery=%s",
            product,
            near_month.code,
            near_month.delivery_date,
        )
        return near_month

    # ------------------------------------------------------------------
    # Snapshots (real-time quote)
    # ------------------------------------------------------------------

    def get_snapshots(self, product: str = "TXF") -> dict:
        """Return the latest snapshot (即時報價) for *product* near-month.

        Returns
        -------
        dict
            Keys: ``code``, ``open``, ``high``, ``low``, ``close``
            (last traded price), ``volume``, ``buy_price``, ``sell_price``,
            ``change_price``, ``change_rate``, ``ts``.
        """
        contract = self.get_contract(product)
        try:
            snaps = _with_retry(self._api.snapshots, [contract])
        except ExecutionError:
            raise
        except Exception as exc:
            raise ExecutionError(f"get_snapshots failed: {exc}") from exc

        if not snaps:
            raise ExecutionError(f"No snapshot returned for {product}")

        snap = snaps[0]
        return {
            "code": snap.code,
            "open": snap.open,
            "high": snap.high,
            "low": snap.low,
            "close": snap.close,
            "volume": snap.volume,
            "total_volume": snap.total_volume,
            "buy_price": snap.buy_price,
            "sell_price": snap.sell_price,
            "change_price": snap.change_price,
            "change_rate": snap.change_rate,
            "ts": snap.ts,
        }

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_order(
        self,
        product: str,
        action: str,
        contracts: int,
        price_type: str = "MKT",
        price: float = 0.0,
        order_type: str = "IOC",
        octype: str = "Auto",
    ) -> dict:
        """Submit a futures order.

        Parameters
        ----------
        product :
            ``"TXF"`` or ``"MXF"``.
        action :
            ``"Buy"`` or ``"Sell"``.
        contracts :
            Number of contracts (口數), must be >= 1.
        price_type :
            ``"MKT"`` (market) or ``"LMT"`` (limit).
        price :
            Limit price; ignored for market orders.
        order_type :
            ``"ROD"`` (default) | ``"IOC"`` | ``"FOK"``.
        octype :
            Open/close type: ``"Auto"`` | ``"New"`` | ``"Cover"``.

        Returns
        -------
        dict
            Keys: ``order_id``, ``status``, ``action``, ``contracts``,
            ``price_type``, ``price``.
        """
        import shioaji as sj

        if contracts < 1:
            raise ValueError(f"contracts must be >= 1, got {contracts}")
        if action not in ("Buy", "Sell"):
            raise ValueError(f"action must be 'Buy' or 'Sell', got {action!r}")
        if price_type not in ("MKT", "LMT", "MKP"):
            raise ValueError(f"price_type must be MKT/LMT/MKP, got {price_type!r}")

        fut_acct = self._futures_account()
        contract = self.get_contract(product)

        order = sj.Order(
            price=price if price_type == "LMT" else 0,
            quantity=contracts,
            action=sj.constant.Action[action],
            price_type=sj.constant.FuturesPriceType[price_type],
            order_type=sj.constant.OrderType[order_type],
            octype=sj.constant.FuturesOCType[octype],
            account=fut_acct,
        )

        logger.info(
            "submit_order: %s %s %d contracts @ %s %s",
            action,
            product,
            contracts,
            price_type,
            f"{price:.0f}" if price_type == "LMT" else "market",
        )
        try:
            trade = _with_retry(self._api.place_order, contract, order)
        except ExecutionError:
            raise
        except Exception as exc:
            raise ExecutionError(f"place_order failed: {exc}") from exc

        status = trade.status
        return {
            "order_id": status.id,
            "status": status.status.value,
            "action": action,
            "contracts": contracts,
            "price_type": price_type,
            "price": price,
        }

    # ------------------------------------------------------------------
    # Position closing helpers
    # ------------------------------------------------------------------

    def close_position(
        self,
        product: str,
        current_direction: str,
        contracts: int,
    ) -> dict:
        """Close *contracts* lots of an existing position.

        Parameters
        ----------
        product :
            ``"TXF"`` or ``"MXF"``.
        current_direction :
            ``"Buy"`` (long) or ``"Sell"`` (short) — direction of the open
            position to close.
        contracts :
            Number of contracts to close.
        """
        # To close a long (Buy), we submit a Sell; and vice versa
        close_action = "Sell" if current_direction == "Buy" else "Buy"
        return self.submit_order(
            product=product,
            action=close_action,
            contracts=contracts,
            price_type="MKT",
            octype="Cover",
        )

    def close_all_positions(self) -> list[dict]:
        """Close all open futures positions at market price.

        Returns
        -------
        list[dict]
            One order result dict per position closed.
        """
        positions = self.get_positions()
        results = []
        for pos in positions:
            logger.info(
                "close_all_positions: closing %s %s × %d",
                pos["direction"],
                pos["code"],
                pos["contracts"],
            )
            # Infer product from code prefix (e.g. "TXFB6" → "TXF")
            code = pos["code"]
            if code.startswith("TXF"):
                product = "TXF"
            elif code.startswith("MXF"):
                product = "MXF"
            else:
                logger.warning("close_all_positions: unknown product code %s — skipping", code)
                continue

            res = self.close_position(
                product=product,
                current_direction=pos["direction"],
                contracts=pos["contracts"],
            )
            results.append(res)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _futures_account(self):
        """Return the futures (futopt) account object.

        Shioaji exposes ``api.futopt_account`` only when the account is
        signed (CA activated).  In simulation mode the property is None, so
        we fall back to scanning ``self._accounts`` for account_type ``H``
        (綜合帳號 / futures-options).
        """
        # Fast path — works in live signed mode
        acct = getattr(self._api, "futopt_account", None)
        if acct is not None:
            return acct

        # Simulation / unsigned fallback: find H-type account manually
        try:
            from shioaji.account import AccountType

            for a in self._accounts:
                if a.account_type == AccountType.H:
                    return a
        except Exception:
            pass

        raise ExecutionError(
            "No futures account (account_type=H) found. "
            "Check that the API credentials include futures permissions."
        )

    # ------------------------------------------------------------------
    # Convenience / backward-compat public API
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """No-op — connection is established in __init__ via _connect()."""

    def place_order(
        self,
        product: str,
        action: str,
        contracts: int,
        price_type: str = "MKT",
        price: float = 0.0,
        order_type: str = "IOC",
        octype: str = "Auto",
    ) -> dict:
        """Alias for :meth:`submit_order` for backward compatibility."""
        return self.submit_order(
            product=product,
            action=action,
            contracts=contracts,
            price_type=price_type,
            price=price,
            order_type=order_type,
            octype=octype,
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by *order_id*.

        Returns
        -------
        bool
            ``True`` if the cancel was accepted by the broker.

        Raises
        ------
        ExecutionError
            If the cancel call fails after all retries.
        """
        try:
            _with_retry(self._api.cancel_order, order_id)
        except ExecutionError:
            raise
        except Exception as exc:
            raise ExecutionError(f"cancel_order failed: {exc}") from exc
        return True
