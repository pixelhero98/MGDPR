"""Utilities for preparing Yahoo Finance OHLCV data."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
from urllib.request import Request, urlopen

import pandas as pd

FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


@dataclass(frozen=True)
class MarketDataResult:
    tickers: List[str]
    ticker_path: Path
    downloaded: List[Path]
    skipped: List[Path]
    failed: List[Tuple[str, str]]


class _SP500TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[str]] = []
        self._in_table = False
        self._table_depth = 0
        self._in_cell = False
        self._current_row: List[str] | None = None
        self._current_cell: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "table" and attrs_dict.get("id") == "constituents":
            self._in_table = True
            self._table_depth = 1
            return
        if not self._in_table:
            return
        if tag == "table":
            self._table_depth += 1
        elif tag == "tr":
            self._current_row = []
        elif tag in {"th", "td"}:
            self._in_cell = True
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._in_table and self._in_cell:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self._in_table:
            return
        if tag in {"th", "td"} and self._in_cell and self._current_row is not None:
            text = " ".join("".join(self._current_cell).split())
            self._current_row.append(text)
            self._in_cell = False
            self._current_cell = []
        elif tag == "tr" and self._current_row:
            self.rows.append(self._current_row)
            self._current_row = None
        elif tag == "table":
            self._table_depth -= 1
            if self._table_depth <= 0:
                self._in_table = False


def normalize_yahoo_symbol(symbol: str) -> str:
    """Convert table symbols to the format expected by Yahoo Finance."""
    return symbol.strip().upper().replace(".", "-")


def default_ticker_path(raw_dir: str | Path, market: str) -> Path:
    return Path(raw_dir).resolve().parent / f"{market}_tickers.csv"


def default_download_range(start: str, end: str) -> Tuple[str, str]:
    """Pad a train/test span so ratio features and next-day labels exist."""
    start_ts = pd.Timestamp(start) - pd.Timedelta(days=45)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=10)
    return start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")


def fetch_sp500_tickers() -> List[str]:
    request = Request(
        SP500_WIKIPEDIA_URL,
        headers={"User-Agent": "MGDPR market data preparation"},
    )
    with urlopen(request, timeout=30) as response:
        markup = response.read().decode("utf-8", errors="replace")

    parser = _SP500TableParser()
    parser.feed(markup)

    if not parser.rows:
        raise RuntimeError("Could not find the S&P 500 constituents table on Wikipedia.")

    header = parser.rows[0]
    try:
        symbol_index = header.index("Symbol")
    except ValueError as exc:
        raise RuntimeError("Could not find the Symbol column in the S&P 500 table.") from exc

    tickers = [
        normalize_yahoo_symbol(row[symbol_index])
        for row in parser.rows[1:]
        if len(row) > symbol_index and row[symbol_index]
    ]
    tickers = [ticker for ticker in tickers if ticker]
    if len(tickers) >= 400:
        return tickers
    raise RuntimeError("Could not find the S&P 500 constituents table on Wikipedia.")


def load_ticker_file(path: str | Path) -> List[str]:
    ticker_path = Path(path)
    tickers: List[str] = []
    with ticker_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            ticker = line.strip().split(",")[0]
            if ticker:
                tickers.append(normalize_yahoo_symbol(ticker))
    if not tickers:
        raise ValueError(f"No tickers found in {ticker_path}.")
    return tickers


def write_ticker_file(tickers: Sequence[str], path: str | Path) -> Path:
    ticker_path = Path(path)
    ticker_path.parent.mkdir(parents=True, exist_ok=True)
    ticker_path.write_text("\n".join(tickers) + "\n", encoding="utf-8")
    return ticker_path


def _flatten_yfinance_columns(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame

    ticker_values = set(frame.columns.get_level_values(-1))
    if ticker in ticker_values:
        return frame.xs(ticker, level=-1, axis=1)

    first_level_values = set(frame.columns.get_level_values(0))
    if ticker in first_level_values:
        return frame.xs(ticker, level=0, axis=1)

    flattened = frame.copy()
    flattened.columns = flattened.columns.get_level_values(0)
    return flattened


def _has_required_date_coverage(path: Path, start: str | None, end: str | None) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False

    frame = pd.read_csv(path, parse_dates=[0], index_col=0)
    if frame.empty:
        return False

    if start is not None and frame.index.min() > pd.Timestamp(start):
        return False
    if end is not None:
        end_bound = pd.Timestamp(end) - pd.Timedelta(days=4)
        if frame.index.max() < end_bound:
            return False
    return True


def _download_one(
    ticker: str,
    raw_dir: Path,
    market: str,
    start: str | None,
    end: str | None,
    force_download: bool,
) -> Tuple[Path, bool]:
    output_path = raw_dir / f"{market}_{ticker}_30Y.csv"
    if not force_download and _has_required_date_coverage(output_path, start, end):
        return output_path, False

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment setup guard
        raise RuntimeError("Install yfinance before downloading market data.") from exc

    frame = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    frame = _flatten_yfinance_columns(frame, ticker)
    frame = frame.rename(columns={"Adj Close": "Close"})

    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{ticker} download is missing columns: {missing}")

    frame = frame[FEATURE_COLUMNS].dropna()
    if frame.empty:
        raise RuntimeError(f"{ticker} download returned no usable OHLCV rows.")

    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    frame.index.name = "Date"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path)
    return output_path, True


def prepare_market_data(
    market: str = "SP500",
    raw_dir: str | Path = "data/raw",
    ticker_path: str | Path | None = None,
    output_ticker_path: str | Path | None = None,
    tickers: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    force_download: bool = False,
    allow_partial: bool = False,
    symbols_only: bool = False,
) -> MarketDataResult:
    market = market.upper()
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    if tickers is not None:
        resolved_tickers = [normalize_yahoo_symbol(ticker) for ticker in tickers if ticker]
        ticker_file = Path(output_ticker_path) if output_ticker_path else default_ticker_path(raw_path, market)
        write_ticker_file(resolved_tickers, ticker_file)
    elif ticker_path is not None:
        ticker_file = Path(ticker_path)
        resolved_tickers = load_ticker_file(ticker_file)
    elif market == "SP500":
        ticker_file = Path(output_ticker_path) if output_ticker_path else default_ticker_path(raw_path, market)
        resolved_tickers = fetch_sp500_tickers()
        write_ticker_file(resolved_tickers, ticker_file)
    else:
        raise ValueError("--ticker_path or explicit tickers are required outside SP500.")

    if not resolved_tickers:
        raise ValueError("No tickers were resolved for market data preparation.")

    downloaded: List[Path] = []
    skipped: List[Path] = []
    failed: List[Tuple[str, str]] = []

    if symbols_only:
        return MarketDataResult(resolved_tickers, ticker_file, downloaded, skipped, failed)

    for ticker in resolved_tickers:
        try:
            path, did_download = _download_one(
                ticker,
                raw_path,
                market,
                start,
                end,
                force_download,
            )
        except Exception as exc:
            if not allow_partial:
                raise RuntimeError(f"Failed to prepare data for {ticker}: {exc}") from exc
            failed.append((ticker, str(exc)))
            continue

        if did_download:
            downloaded.append(path)
        else:
            skipped.append(path)

    if failed and len(failed) == len(resolved_tickers):
        raise RuntimeError("All ticker downloads failed.")

    return MarketDataResult(resolved_tickers, ticker_file, downloaded, skipped, failed)
