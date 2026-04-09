"""SQLAlchemy table definitions — single source of truth for the DB schema.

Tables:
  strategy_candidates  — output of Strategy Researcher
  backtest_results     — output of Backtester
  approved_orders      — output of Risk Manager
  trades               — filled orders written by Executor
  positions            — current holdings, updated by Executor
  order_errors         — failed order attempts
  risk_events          — risk limit breaches logged by Risk Manager / Monitor
  system_state         — key-value store for global flags (e.g. trading_halted)
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class StrategyCandidate(Base):
    __tablename__ = "strategy_candidates"

    id = Column(String, primary_key=True)
    factor_name = Column(String, nullable=False)
    params = Column(JSON, nullable=False)
    ic_score = Column(Float)
    universe = Column(String)
    status = Column(String, default="pending")  # pending | approved | rejected
    created_at = Column(DateTime)
    updated_at = Column(DateTime)


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(String, nullable=False)
    sharpe = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    cagr = Column(Float)
    calmar = Column(Float)
    oos_windows = Column(Integer)
    verdict = Column(String)  # approved | rejected
    created_at = Column(DateTime)


class ApprovedOrder(Base):
    __tablename__ = "approved_orders"

    id = Column(String, primary_key=True)
    strategy_id = Column(String)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)  # buy | sell
    qty = Column(Integer, nullable=False)
    order_type = Column(String, default="market")
    limit_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    status = Column(String, default="pending")  # pending | submitted | filled | cancelled | error
    created_at = Column(DateTime)


class Trade(Base):
    __tablename__ = "trades"

    id = Column(String, primary_key=True)
    order_id = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    qty = Column(Integer, nullable=False)
    fill_price = Column(Float, nullable=False)
    broker = Column(String)
    filled_at = Column(DateTime)


class Position(Base):
    __tablename__ = "positions"

    symbol = Column(String, primary_key=True)
    qty = Column(Integer, nullable=False)
    avg_cost = Column(Float, nullable=False)
    broker = Column(String)
    updated_at = Column(DateTime)


class OrderError(Base):
    __tablename__ = "order_errors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String)
    symbol = Column(String)
    error = Column(Text)
    created_at = Column(DateTime)


class RiskEvent(Base):
    __tablename__ = "risk_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    rule = Column(String, nullable=False)
    severity = Column(String)
    message = Column(Text)
    created_at = Column(DateTime)


class SystemState(Base):
    __tablename__ = "system_state"

    key = Column(String, primary_key=True)
    value = Column(Text)
    updated_at = Column(DateTime)
