import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


def utc_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_tail(path: Path, max_bytes: int = 200_000) -> str:
    if not path.exists():
        return ""
    try:
        data = path.read_bytes()
        if len(data) <= max_bytes:
            return data.decode("utf-8", errors="replace")
        return data[-max_bytes:].decode("utf-8", errors="replace")
    except Exception:
        return ""


def scan_patterns(text: str, patterns: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pat in patterns:
        counts[pat] = len(re.findall(pat, text, flags=re.IGNORECASE | re.MULTILINE))
    return counts


def mt5_snapshot_from_python() -> dict:
    """Runs an in-process MT5 snapshot via python -c to avoid shell quoting problems."""
    code = r"""
import os
from dotenv import load_dotenv
load_dotenv('config/secrets.env')

try:
    import MetaTrader5 as mt5
except Exception as e:
    print('{"ok": false, "error": "mt5_import_failed", "detail": "%s"}' % (str(e).replace('"','\\"')))
    raise SystemExit(0)

login = os.getenv('MT5_LOGIN')
pwd = os.getenv('MT5_PASSWORD')
srv = os.getenv('MT5_SERVER')

ok = False
try:
    ok = mt5.initialize(login=int(login), password=pwd, server=srv)
except Exception:
    ok = mt5.initialize()

if not ok:
    print('{"ok": false, "error": "mt5_initialize_failed", "last_error": "%s"}' % (str(mt5.last_error()).replace('"','\\"')))
    raise SystemExit(0)

ai = mt5.account_info()
positions = mt5.positions_get()

out = {
  "ok": True,
  "server": getattr(ai, 'server', None),
  "trade_mode": getattr(ai, 'trade_mode', None),
  "balance": getattr(ai, 'balance', None),
  "equity": getattr(ai, 'equity', None),
  "profit": getattr(ai, 'profit', None),
  "margin_free": getattr(ai, 'margin_free', None),
  "positions_count": 0 if positions is None else len(positions),
  "positions": []
}

for p in (positions or []):
    out["positions"].append({
        "ticket": p.ticket,
        "symbol": p.symbol,
        "type": p.type,
        "volume": p.volume,
        "price_open": p.price_open,
        "price_current": getattr(p, 'price_current', None),
        "profit": p.profit,
        "swap": getattr(p, 'swap', 0.0),
        "magic": getattr(p, 'magic', 0),
        "comment": getattr(p, 'comment', '')
    })

mt5.shutdown()
import json
print(json.dumps(out))
"""
    exe = Path(".venv") / "Scripts" / "python.exe"
    proc = subprocess.run([str(exe), "-c", code], capture_output=True, text=True)
    raw = (proc.stdout or "").strip().splitlines()[-1] if proc.stdout else ""
    try:
        return json.loads(raw) if raw else {"ok": False, "error": "no_output"}
    except Exception:
        return {"ok": False, "error": "invalid_json", "raw": raw, "stderr": proc.stderr}


def decisions_tail_counts(decisions_dir: Path, tail_lines: int = 5000) -> dict:
    latest = None
    for f in sorted(decisions_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        latest = f
        break

    if latest is None:
        return {"file": None, "events": 0, "action_counts": {}, "reason_counts": {}}

    try:
        lines = latest.read_text(encoding="utf-8", errors="replace").splitlines()[-tail_lines:]
    except Exception:
        return {"file": str(latest), "events": 0, "action_counts": {}, "reason_counts": {}}

    action_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    events = 0

    for line in lines:
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        events += 1
        action = obj.get("action")
        if action:
            action_counts[action] = action_counts.get(action, 0) + 1
        reasons = (((obj.get("decision") or {}).get("reasons")) or [])
        if isinstance(reasons, list):
            for r in reasons:
                if not isinstance(r, str):
                    continue
                reason_counts[r] = reason_counts.get(r, 0) + 1

    # Keep only top reasons
    top_reasons = dict(sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:15])
    top_actions = dict(sorted(action_counts.items(), key=lambda kv: kv[1], reverse=True)[:15])

    return {"file": str(latest), "events": events, "action_counts": top_actions, "reason_counts": top_reasons}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=int, default=60)
    parser.add_argument("--interval", type=int, default=60, help="Seconds between snapshots")
    parser.add_argument("--symbol", type=str, default="XAUUSD")
    args = parser.parse_args()

    # Ensure dotenv load (some parts rely on it)
    load_dotenv("config/secrets.env")

    root = Path.cwd()
    logs_dir = root / "logs"
    decisions_dir = logs_dir / "decisions"
    monitor_dir = logs_dir / "monitor"
    safe_mkdir(monitor_dir)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = monitor_dir / f"bot_{run_id}.out"
    err_path = monitor_dir / f"bot_{run_id}.err"
    report_path = monitor_dir / f"report_{run_id}.json"
    summary_path = monitor_dir / f"summary_{run_id}.txt"

    # Start bot
    exe = root / ".venv" / "Scripts" / "python.exe"
    bot = subprocess.Popen(
        [str(exe), "run_bot.py"],
        cwd=str(root),
        stdout=open(out_path, "w", encoding="utf-8", errors="replace"),
        stderr=open(err_path, "w", encoding="utf-8", errors="replace"),
        creationflags=0,
    )

    start = time.time()
    end = start + (args.minutes * 60)

    patterns = [r"\bFATAL\b", r"\bTraceback\b", r"\bException\b", r"\bERROR\b", r"\bWARN\b"]

    samples: list[dict] = []
    issues: list[dict] = []

    last_decisions = None

    try:
        while time.time() < end:
            snap = {
                "ts": utc_ts(),
                "iso": iso_utc(),
                "mt5": mt5_snapshot_from_python(),
            }

            out_tail = read_tail(out_path)
            err_tail = read_tail(err_path)
            snap["stdout_counts"] = scan_patterns(out_tail, patterns)
            snap["stderr_counts"] = scan_patterns(err_tail, patterns)

            dec = decisions_tail_counts(decisions_dir)
            snap["decisions"] = dec

            # Detect increases in error signals
            if samples:
                prev = samples[-1]
                for key in ["stdout_counts", "stderr_counts"]:
                    for pat, val in snap[key].items():
                        if val > prev.get(key, {}).get(pat, 0):
                            issues.append({"ts": snap["ts"], "type": "log_pattern", "stream": key, "pattern": pat, "count": val})

            # Detect decisions file changes
            if last_decisions and dec.get("file") == last_decisions.get("file"):
                if dec.get("events", 0) < last_decisions.get("events", 0):
                    issues.append({"ts": snap["ts"], "type": "decisions_reset", "file": dec.get("file")})
            last_decisions = dec

            samples.append(snap)
            time.sleep(max(5, args.interval))

    finally:
        if bot.poll() is None:
            bot.terminate()
            try:
                bot.wait(timeout=10)
            except Exception:
                bot.kill()

    duration_s = time.time() - start

    # Final aggregates
    final_dec = decisions_tail_counts(decisions_dir)
    last = samples[-1] if samples else {}
    first = samples[0] if samples else {}

    report = {
        "run_id": run_id,
        "started_utc": first.get("iso"),
        "ended_utc": last.get("iso"),
        "duration_seconds": duration_s,
        "bot_stdout": str(out_path),
        "bot_stderr": str(err_path),
        "final_decisions": final_dec,
        "issues": issues[:200],
        "samples": samples,
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Human summary
    mt5_first = (first.get("mt5") or {})
    mt5_last = (last.get("mt5") or {})
    lines = []
    lines.append(f"Run: {run_id}")
    lines.append(f"Duration: {int(duration_s)}s")
    lines.append(f"Bot output: {out_path}")
    lines.append(f"Bot errors: {err_path}")
    if mt5_first.get("ok") and mt5_last.get("ok"):
        lines.append(f"MT5 server: {mt5_last.get('server')}")
        lines.append(f"Equity: {mt5_first.get('equity')} -> {mt5_last.get('equity')}")
        lines.append(f"Open positions: {mt5_first.get('positions_count')} -> {mt5_last.get('positions_count')}")
    else:
        lines.append(f"MT5 snapshot not ok: first={mt5_first.get('error')} last={mt5_last.get('error')}")

    lines.append("\nTop decision actions (tail):")
    for k, v in (final_dec.get("action_counts") or {}).items():
        lines.append(f"- {k}: {v}")

    lines.append("\nTop decision reasons (tail):")
    for k, v in (final_dec.get("reason_counts") or {}).items():
        lines.append(f"- {k}: {v}")

    if issues:
        lines.append("\nIssues detected:")
        for it in issues[:25]:
            lines.append(f"- {it}")
    else:
        lines.append("\nIssues detected: none (by pattern scan)")

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(str(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
