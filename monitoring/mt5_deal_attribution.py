"""Forensic helper: attribute MT5 closes to an EA/bot.

Reads MT5 history deals within a time window and prints deal metadata
(magic/comment/reason/profit), allowing you to confirm whether a burst of
closes came from this bot (e.g., comment "Aether FastClose") vs manual/other EA.

Usage examples (GMT+5 default):
  python monitoring/mt5_deal_attribution.py --from "2025-12-18 19:13:45" --to "2025-12-18 19:14:10" --symbol XAUUSD
  python monitoring/mt5_deal_attribution.py --from "2025-12-18 19:13:45+05:00" --to "2025-12-18 19:14:10+05:00" --symbol XAUUSD

If MT5 isn't connected, launch the MT5 terminal first and login to the
same account, then re-run.
"""

from __future__ import annotations

import argparse
import datetime as dt
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional


def _parse_dt(value: str, default_tz: dt.tzinfo) -> dt.datetime:
    value = value.strip()
    # Accept either:
    # - "YYYY-MM-DD HH:MM:SS" (assumed default_tz)
    # - ISO strings including offset, e.g. "2025-12-18T19:13:54+05:00" or "2025-12-18 19:13:54+05:00"
    if "T" not in value and "+" not in value and "-" in value[:10] and value.count(":") >= 1:
        # naive-ish local time
        parsed = dt.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        return parsed.replace(tzinfo=default_tz)

    # Normalize space separator to 'T' for fromisoformat
    normalized = value.replace(" ", "T")
    parsed = dt.datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=default_tz)
    return parsed


@dataclass(frozen=True)
class DealRow:
    time: dt.datetime
    ticket: int
    order: int
    position_id: int
    symbol: str
    entry: int
    type: int
    volume: float
    price: float
    profit: float
    commission: float
    swap: float
    magic: int
    comment: str
    reason: int


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_dt_from_mt5_seconds(seconds: object, tz: dt.tzinfo) -> dt.datetime:
    sec = _safe_int(seconds)
    return dt.datetime.fromtimestamp(sec, tz=dt.timezone.utc).astimezone(tz)


def _to_deal_rows(deals: Iterable[object], tz: dt.tzinfo) -> list[DealRow]:
    rows: list[DealRow] = []
    for d in deals:
        rows.append(
            DealRow(
                time=_as_dt_from_mt5_seconds(getattr(d, "time", 0), tz),
                ticket=_safe_int(getattr(d, "ticket", 0)),
                order=_safe_int(getattr(d, "order", 0)),
                position_id=_safe_int(getattr(d, "position_id", 0)),
                symbol=str(getattr(d, "symbol", "")),
                entry=_safe_int(getattr(d, "entry", 0)),
                type=_safe_int(getattr(d, "type", 0)),
                volume=_safe_float(getattr(d, "volume", 0.0)),
                price=_safe_float(getattr(d, "price", 0.0)),
                profit=_safe_float(getattr(d, "profit", 0.0)),
                commission=_safe_float(getattr(d, "commission", 0.0)),
                swap=_safe_float(getattr(d, "swap", 0.0)),
                magic=_safe_int(getattr(d, "magic", 0)),
                comment=str(getattr(d, "comment", "")),
                reason=_safe_int(getattr(d, "reason", 0)),
            )
        )
    rows.sort(key=lambda r: (r.time, r.ticket, r.order))
    return rows


def _print_rows(rows: list[DealRow]) -> None:
    if not rows:
        print("No deals found in the requested window.")
        return

    print(f"Deals found: {len(rows)}")
    header = (
        "time",
        "symbol",
        "ticket",
        "pos_id",
        "order",
        "type",
        "entry",
        "vol",
        "price",
        "profit",
        "magic",
        "comment",
        "reason",
    )
    print("\t".join(header))
    for r in rows:
        print(
            "\t".join(
                [
                    r.time.strftime("%Y-%m-%d %H:%M:%S%z"),
                    r.symbol,
                    str(r.ticket),
                    str(r.position_id),
                    str(r.order),
                    str(r.type),
                    str(r.entry),
                    f"{r.volume:.2f}",
                    f"{r.price:.2f}",
                    f"{r.profit:.2f}",
                    str(r.magic),
                    (r.comment or "").replace("\t", " ").replace("\n", " "),
                    str(r.reason),
                ]
            )
        )


def _print_summary(rows: list[DealRow], symbol: Optional[str]) -> None:
    if not rows:
        return

    if symbol:
        rows = [r for r in rows if r.symbol.upper() == symbol.upper()]
        if not rows:
            return

    by_magic = Counter(r.magic for r in rows)
    by_comment = Counter((r.comment or "").strip() for r in rows)

    print("\nSummary")
    print(f"Total profit (incl. commission+swap not added): {sum(r.profit for r in rows):.2f}")

    print("\nTop magic numbers:")
    for magic, count in by_magic.most_common(8):
        print(f"  magic={magic}: {count} deals")

    print("\nTop comments:")
    for comment, count in by_comment.most_common(8):
        label = comment if comment else "<empty>"
        print(f"  {label!r}: {count} deals")

    # Flag our known close comment
    aether = sum(1 for r in rows if (r.comment or "").strip() == "Aether FastClose")
    if aether:
        print(f"\nDetected {aether} deal(s) with comment 'Aether FastClose' -> bot-issued fast closes.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Attribute MT5 closes to EA/bot via deal metadata")
    parser.add_argument("--from", dest="from_dt", required=True, help="Start time (e.g. '2025-12-18 19:13:45')")
    parser.add_argument("--to", dest="to_dt", required=True, help="End time (e.g. '2025-12-18 19:14:10')")
    parser.add_argument("--tz", default="+05:00", help="Timezone offset for naive inputs. Default: +05:00")
    parser.add_argument("--symbol", default=None, help="Optional symbol filter (e.g. XAUUSD)")
    args = parser.parse_args()

    # Parse tz like +05:00 or -03:30
    tz_s = args.tz.strip()
    sign = 1
    if tz_s.startswith("-"):
        sign = -1
        tz_s = tz_s[1:]
    elif tz_s.startswith("+"):
        tz_s = tz_s[1:]
    hh, mm = tz_s.split(":")
    offset = dt.timedelta(hours=int(hh) * sign, minutes=int(mm) * sign)
    default_tz = dt.timezone(offset)

    start = _parse_dt(args.from_dt, default_tz)
    end = _parse_dt(args.to_dt, default_tz)
    if end <= start:
        raise SystemExit("--to must be after --from")

    try:
        import MetaTrader5 as mt5  # type: ignore
    except Exception as e:
        print("Failed to import MetaTrader5. Is the 'MetaTrader5' package installed in this venv?")
        print(f"Import error: {e}")
        return 2

    if not mt5.initialize():
        print("mt5.initialize() failed.")
        print("- Ensure the MT5 terminal is running on this machine")
        print("- Ensure you're logged in to the account")
        print("- Ensure Algo Trading is enabled")
        print("- Then re-run this script")
        return 3

    # The MetaTrader5 Python API generally expects *naive* datetimes that are interpreted
    # in the terminal's local timezone. To avoid off-by-offset mistakes, we try:
    #  1) local-naive (based on --tz)
    #  2) UTC-naive fallback (some setups interpret inputs as UTC)
    start_local_naive = start.astimezone(default_tz).replace(tzinfo=None)
    end_local_naive = end.astimezone(default_tz).replace(tzinfo=None)
    deals = mt5.history_deals_get(start_local_naive, end_local_naive)
    if deals is not None and len(deals) == 0:
        start_utc_naive = start.astimezone(dt.timezone.utc).replace(tzinfo=None)
        end_utc_naive = end.astimezone(dt.timezone.utc).replace(tzinfo=None)
        deals = mt5.history_deals_get(start_utc_naive, end_utc_naive)
    if deals is None:
        print("history_deals_get returned None (no history or terminal not ready).")
        mt5.shutdown()
        return 4

    rows = _to_deal_rows(deals, default_tz)
    if args.symbol:
        rows = [r for r in rows if r.symbol.upper() == args.symbol.upper()]

    _print_rows(rows)
    _print_summary(rows, args.symbol)

    mt5.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
