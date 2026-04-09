"""Global configuration via pydantic-settings. Values loaded from .env."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    DB_PATH: str = "data/db/trading.db"
    MARKET_DB_PATH: str = "data/db/market_data.db"

    # ------------------------------------------------------------------
    # Alpaca — broker + data
    # ------------------------------------------------------------------
    ALPACA_API_KEY: str = ""
    ALPACA_SECRET_KEY: str = ""
    ALPACA_PAPER: bool = True

    # Data feed: "iex" = free (IEX exchange only, may lag);
    #            "sip" = consolidated tape (requires Alpaca paid plan)
    ALPACA_DATA_FEED: str = "iex"

    # ------------------------------------------------------------------
    # Taiwan broker — "shioaji" | "fugle"
    # ------------------------------------------------------------------
    TW_BROKER: str = "shioaji"

    # Simulation mode (True = paper trading, no cert required)
    SHIOAJI_SIMULATION: bool = True

    SHIOAJI_API_KEY: str = ""
    SHIOAJI_SECRET_KEY: str = ""

    # Certificate — required only for live trading (SHIOAJI_SIMULATION=false)
    SHIOAJI_CERT_PATH: str | None = None
    SHIOAJI_CERT_PASSWORD: str | None = None
    SHIOAJI_PERSON_ID: str | None = None

    FUGLE_API_KEY: str = ""
    FUGLE_SECRET_KEY: str = ""

    # ------------------------------------------------------------------
    # Risk limits
    # ------------------------------------------------------------------
    MAX_PORTFOLIO_DRAWDOWN: float = 0.10  # 10% — halt trading
    WARN_PORTFOLIO_DRAWDOWN: float = 0.07  # 7% — alert only
    MAX_SINGLE_POSITION_PCT: float = 0.05  # 5% of equity per symbol
    MAX_SECTOR_CONCENTRATION: float = 0.30

    # ------------------------------------------------------------------
    # Execution cost model
    # ------------------------------------------------------------------
    SLIPPAGE_MODEL: str = "fixed"  # "fixed" | "linear_impact"
    SLIPPAGE_BPS: float = 10.0

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------
    LINE_NOTIFY_TOKEN: str = ""
    SLACK_WEBHOOK_URL: str = ""
    ALERT_EMAIL: str = ""

    # ------------------------------------------------------------------
    # LINE Messaging API (Push Message)
    # ------------------------------------------------------------------
    LINE_CHANNEL_ACCESS_TOKEN: str = ""  # long-lived token from Developers Console
    LINE_CHANNEL_SECRET: str = ""  # webhook verification (not used for push)
    LINE_USER_ID: str = ""  # recipient "U..." id

    # ------------------------------------------------------------------
    # SMTP (email notifications)
    # ------------------------------------------------------------------
    SMTP_HOST: str = ""  # e.g. smtp.gmail.com
    SMTP_PORT: int = 587  # 587 = STARTTLS, 465 = SSL
    SMTP_USER: str = ""  # sender address / login
    SMTP_PASS: str = ""  # password or app-specific password
    NOTIFY_EMAIL: str = ""  # recipient address

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------
    DASHBOARD_BACKEND: str = "rich"  # "rich" | "streamlit"

    # ------------------------------------------------------------------
    # Backtest parameters
    # ------------------------------------------------------------------
    WF_IN_SAMPLE_BARS: int = 252  # ~1 trading year
    WF_OUT_OF_SAMPLE_BARS: int = 63  # ~1 quarter
    WF_MIN_OOS_WINDOWS: int = 3
    INIT_CASH: float = 1_000_000.0


settings = Settings()
