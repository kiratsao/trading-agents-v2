"""Intraday (當沖) strategies for TAIFEX futures.

TODO: Implement MeanReversionIntradayStrategy once Shioaji minute-bar
      data is available via tw_futures.data.fetcher.TaifexFetcher.fetch_minute().
"""

from .mean_reversion_intraday import IntradaySignal, MeanReversionIntradayStrategy

__all__ = ["IntradaySignal", "MeanReversionIntradayStrategy"]
