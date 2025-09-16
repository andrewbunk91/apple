"""Utility script to download historical stock data with yfinance."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


def download_stock_history(
    ticker: str,
    start: str,
    end: Optional[str],
    interval: str,
) -> pd.DataFrame:
    """Download historical stock data for *ticker* using yfinance.

    Parameters
    ----------
    ticker:
        The stock ticker symbol (e.g. ``"AAPL"``).
    start:
        Start date in the format ``YYYY-MM-DD``.
    end:
        Optional end date. When omitted, yfinance will use today's date.
    interval:
        Data sampling frequency (``"1d"`` for daily prices, ``"1wk"`` for weekly, ...).

    Returns
    -------
    pandas.DataFrame
        The downloaded OHLCV data frame.
    """

    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )

    if data.empty:
        raise RuntimeError(
            "No data was returned. Check the ticker symbol and provided date range."
        )

    return data


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the downloader script."""

    parser = argparse.ArgumentParser(
        description="Download historical stock data using yfinance",
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        default="AAPL",
        help="Ticker symbol to download (default: %(default)s)",
    )
    parser.add_argument(
        "--start",
        default="2010-01-01",
        help="Start date in YYYY-MM-DD format (default: %(default)s)",
    )
    parser.add_argument(
        "--end",
        default="2023-01-01",
        help="Optional end date in YYYY-MM-DD format (default: %(default)s)",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Data interval accepted by yfinance (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("apple_stock_data_2023.csv"),
        help="Destination CSV file (default: %(default)s)",
    )

    return parser.parse_args()


def main() -> None:
    """Entrypoint for the downloader script."""

    args = parse_args()
    data = download_stock_history(args.ticker, args.start, args.end, args.interval)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path)

    print(f"Saved {len(data)} rows to {output_path.resolve()}")


if __name__ == "__main__":
    main()
