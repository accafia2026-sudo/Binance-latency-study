"""
REST latency probe against Binance.

Measures round-trip time of a signed GET /api/v3/time request every 500 ms.
Writes to data/rest_latency.csv.

Usage:
    python probe_rest.py
"""
import asyncio
import aiohttp
import time
import csv
import os
from datetime import datetime, timezone

REST_URL = "https://api.binance.com/api/v3/time"
DATA_DIR = os.path.expanduser("~/latency-research/data")
LOG_FILE = os.path.join(DATA_DIR, "rest_latency.csv")
INTERVAL = 0.5  # seconds


async def probe_once(session):
    try:
        t0 = time.perf_counter()
        async with session.get(
            REST_URL, timeout=aiohttp.ClientTimeout(total=5)
        ) as resp:
            await resp.read()
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0
    except Exception:
        return None


async def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    write_header = not os.path.exists(LOG_FILE)

    async with aiohttp.ClientSession() as session:
        with open(LOG_FILE, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp_utc", "timestamp_unix_ms", "latency_ms"])

            print(f"Logging to {LOG_FILE}. Ctrl+C to stop.")
            while True:
                ts_unix = int(time.time() * 1000)
                lat = await probe_once(session)
                ts_utc = datetime.now(timezone.utc).isoformat()
                if lat is not None:
                    w.writerow([ts_utc, ts_unix, f"{lat:.3f}"])
                    f.flush()
                    print(f"{ts_utc}  {lat:.2f} ms")
                else:
                    print(f"{ts_utc}  FAILED")
                await asyncio.sleep(INTERVAL)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")