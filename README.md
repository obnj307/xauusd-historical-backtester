# XAUUSD M30 Historical Backtester — Phase 1

This starter validates imported historical M30 bars; **it does not calculate trading signals, PnL or win rate yet**.

## Setup (works from Android GitHub browser)

1. Create a **private** GitHub repository, e.g. `xauusd-historical-backtester`.
2. Upload `backtest.py` and `.github/workflows/audit.yml` preserving paths.
3. Obtain historical **XAUUSD M30** OHLC data from a provider that permits download. Broker data is preferred. Verify the timezone and symbol contract specification; do not mix providers without checking.
4. Upload a CSV to `data/XAUUSD_M30.csv` with the exact headers:

```csv
time,open,high,low,close,volume
2026-01-05T00:00:00Z,4400.0,4401.0,4399.0,4400.5,100
2026-01-05T00:30:00Z,4400.5,4402.0,4400.0,4401.0,120
```

These are **format examples, not actual historical prices**. `volume` is optional. `time` must represent the UTC opening time of each bar. If a source supplies broker-local timestamps, convert them to UTC first.

5. Go to **Actions → XAUUSD M30 Data Audit → Run workflow**.
6. Read the job log and download `xauusd-data-audit` artifact containing `audit.json`.

## Notes

- The script sorts bars, validates OHLC, checks duplicate timestamps, reports first/last bar, counts bars by year and identifies >30-minute gaps. Market closures are not necessarily errors.
- GitHub web upload has file size limits; large files may require splitting by year or using Git LFS. This starter expects a single CSV file.
- A data source is **not included**. We must obtain lawful and reliable data first.
- Phase 2 will port the Pine v6.9.20 signal calculation and exact execution rules and verify 2026 trade-by-trade parity against TradingView before any historical performance claims.
