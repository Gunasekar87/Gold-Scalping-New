import argparse
import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv


def _dt(ts: int) -> str:
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except Exception:
        return str(ts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickets", nargs="+", required=True, help="MT5 position tickets / position_id values")
    ap.add_argument("--hours", type=int, default=12, help="How far back to query history")
    args = ap.parse_args()

    ticket_set = set()
    for t in args.tickets:
        try:
            ticket_set.add(int(t))
        except Exception:
            pass

    load_dotenv("config/secrets.env")

    import MetaTrader5 as mt5

    login = os.getenv("MT5_LOGIN")
    pwd = os.getenv("MT5_PASSWORD")
    srv = os.getenv("MT5_SERVER")

    ok = False
    try:
        if login and pwd and srv:
            ok = mt5.initialize(login=int(login), password=pwd, server=srv)
        else:
            ok = mt5.initialize()
    except Exception:
        ok = mt5.initialize()

    if not ok:
        print(f"MT5 initialize failed: {mt5.last_error()}")
        return 2

    utc_to = datetime.now(timezone.utc)
    utc_from = utc_to - timedelta(hours=args.hours)

    deals = mt5.history_deals_get(utc_from, utc_to)
    orders = mt5.history_orders_get(utc_from, utc_to)

    if deals is None:
        print(f"history_deals_get returned None: {mt5.last_error()}")
        mt5.shutdown()
        return 3

    # Filter deals by position_id
    filtered = [d for d in deals if getattr(d, "position_id", None) in ticket_set]
    filtered.sort(key=lambda d: getattr(d, "time", 0))

    print(f"Queried deals: {len(deals)} total; matched position_ids: {len(filtered)}")

    # Print the key fields we need to diagnose closes
    for d in filtered:
        print(
            "DEAL | "
            f"time_utc={_dt(getattr(d,'time',0))} | "
            f"position_id={getattr(d,'position_id',None)} | "
            f"deal={getattr(d,'ticket',None)} | "
            f"order={getattr(d,'order',None)} | "
            f"symbol={getattr(d,'symbol',None)} | "
            f"type={getattr(d,'type',None)} | "
            f"entry={getattr(d,'entry',None)} | "
            f"volume={getattr(d,'volume',None)} | "
            f"price={getattr(d,'price',None)} | "
            f"profit={getattr(d,'profit',None)} | "
            f"commission={getattr(d,'commission',None)} | "
            f"swap={getattr(d,'swap',None)} | "
            f"magic={getattr(d,'magic',None)} | "
            f"reason={getattr(d,'reason',None)} | "
            f"comment={getattr(d,'comment',None)}"
        )

    # Optional: show matching orders too (often has clearer state)
    if orders is not None:
        ord_filtered = [o for o in orders if getattr(o, "position_id", None) in ticket_set or getattr(o, "ticket", None) in ticket_set]
        ord_filtered.sort(key=lambda o: getattr(o, "time_setup", 0))
        print(f"\nQueried orders: {len(orders)} total; matched: {len(ord_filtered)}")
        for o in ord_filtered:
            print(
                "ORDER | "
                f"time_setup_utc={_dt(getattr(o,'time_setup',0))} | "
                f"position_id={getattr(o,'position_id',None)} | "
                f"ticket={getattr(o,'ticket',None)} | "
                f"symbol={getattr(o,'symbol',None)} | "
                f"type={getattr(o,'type',None)} | "
                f"state={getattr(o,'state',None)} | "
                f"volume={getattr(o,'volume_current',None)} | "
                f"price={getattr(o,'price_current',None)} | "
                f"magic={getattr(o,'magic',None)} | "
                f"comment={getattr(o,'comment',None)}"
            )

    mt5.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
