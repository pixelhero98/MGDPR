"""Command line entrypoint for preparing MGDPR market data."""

from __future__ import annotations

import argparse

from .market_data import prepare_market_data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OHLCV CSV files for MGDPR.")
    parser.add_argument("--market", default="SP500")
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--ticker_path", default=None)
    parser.add_argument("--output_ticker_path", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument("--symbols_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = prepare_market_data(
        market=args.market,
        raw_dir=args.raw_dir,
        ticker_path=args.ticker_path,
        output_ticker_path=args.output_ticker_path,
        start=args.start,
        end=args.end,
        force_download=args.force_download,
        allow_partial=args.allow_partial,
        symbols_only=args.symbols_only,
    )

    print(f"Ticker file: {result.ticker_path}")
    print(f"Tickers: {len(result.tickers)}")
    if args.symbols_only:
        return
    print(f"Downloaded: {len(result.downloaded)}")
    print(f"Skipped existing: {len(result.skipped)}")
    if result.failed:
        print(f"Failed: {len(result.failed)}")
        for ticker, reason in result.failed[:10]:
            print(f"  {ticker}: {reason}")


if __name__ == "__main__":
    main()
