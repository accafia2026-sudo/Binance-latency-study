"""
Multi-stream WebSocket latency recorder for Binance BTCUSDT.

Streams:
  - Spot trades        wss://stream.binance.com:9443/ws/btcusdt@trade
  - Futures BBO        wss://fstream.binance.com/ws/btcusdt@bookTicker
  - Futures trades     wss://fstream.binance.com/ws/btcusdt@trade

Each message is timestamped immediately after socket receipt (recv_ms)
and the exchange event time (event_time_ms) is extracted from the
payload. Latency = recv_ms - event_time_ms.

Writes gzipped CSV to ~/latency-research/data/.

Usage:
    python probe_ws.py
"""
import asyncio
import websockets
import json
import gzip
import os
import time
from datetime import datetime, timezone

# --- Endpoints (single-stream URL format, most reliable for futures) ---
SPOT_URL      = "wss://stream.binance.com:9443/ws/btcusdt@trade"
FUT_BBO_URL   = "wss://fstream.binance.com/ws/btcusdt@bookTicker"
FUT_TRADE_URL = "wss://fstream.binance.com/ws/btcusdt@trade"

DATA_DIR = os.path.expanduser("~/latency-research/data")
os.makedirs(DATA_DIR, exist_ok=True)

SPOT_FILE = os.path.join(DATA_DIR, "btc_spot_trades.csv.gz")
FUT_BBO   = os.path.join(DATA_DIR, "btc_fut_bbo.csv.gz")
FUT_TRADE = os.path.join(DATA_DIR, "btc_fut_trades.csv.gz")

SPOT_HDR = ["recv_utc", "recv_ms", "event_time_ms", "latency_ms",
            "price", "qty", "is_buyer_maker"]
BBO_HDR  = ["recv_utc", "recv_ms", "event_time_ms", "transact_time_ms",
            "latency_ms", "bid_px", "bid_qty", "ask_px", "ask_qty"]
TRD_HDR  = ["recv_utc", "recv_ms", "event_time_ms", "trade_time_ms",
            "latency_ms", "price", "qty", "is_buyer_maker"]

FLUSH_EVERY = 25


def open_gz(path, header):
    new = not os.path.exists(path)
    f = gzip.open(path, "at", newline="")
    if new:
        f.write(",".join(header) + "\n")
        f.flush()
    return f


async def spot_trades():
    f = open_gz(SPOT_FILE, SPOT_HDR)
    print(f"[SPOT]     {SPOT_URL}")
    msg = 0
    async for ws in websockets.connect(
        SPOT_URL, ping_interval=20, ping_timeout=20, max_size=2**22
    ):
        try:
            async for raw in ws:
                recv_ms = time.time() * 1000
                recv_utc = datetime.now(timezone.utc).isoformat()
                try:
                    d = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                et = d.get("E"); p = d.get("p")
                q = d.get("q"); m = d.get("m")
                if et and p and q:
                    lat = recv_ms - et
                    f.write(f"{recv_utc},{recv_ms:.3f},{et},{lat:.3f},"
                            f"{p},{q},{m}\n")
                    msg += 1
                    if msg % FLUSH_EVERY == 0:
                        f.flush()
        except websockets.ConnectionClosed:
            print("[SPOT] reconnect")
            continue


async def fut_bbo():
    f = open_gz(FUT_BBO, BBO_HDR)
    print(f"[FUT-BBO]  {FUT_BBO_URL}")
    msg = 0
    async for ws in websockets.connect(
        FUT_BBO_URL, ping_interval=20, ping_timeout=20, max_size=2**22
    ):
        try:
            async for raw in ws:
                recv_ms = time.time() * 1000
                recv_utc = datetime.now(timezone.utc).isoformat()
                try:
                    d = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                et = d.get("E"); tt = d.get("T")
                bp = d.get("b"); bq = d.get("B")
                ap = d.get("a"); aq = d.get("A")
                if et and bp and ap:
                    lat = recv_ms - et
                    f.write(f"{recv_utc},{recv_ms:.3f},{et},{tt},{lat:.3f},"
                            f"{bp},{bq},{ap},{aq}\n")
                    msg += 1
                    if msg % FLUSH_EVERY == 0:
                        f.flush()
        except websockets.ConnectionClosed:
            print("[FUT-BBO] reconnect")
            continue


async def fut_trades():
    f = open_gz(FUT_TRADE, TRD_HDR)
    print(f"[FUT-TRD]  {FUT_TRADE_URL}")
    msg = 0
    async for ws in websockets.connect(
        FUT_TRADE_URL, ping_interval=20, ping_timeout=20, max_size=2**22
    ):
        try:
            async for raw in ws:
                recv_ms = time.time() * 1000
                recv_utc = datetime.now(timezone.utc).isoformat()
                try:
                    d = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                tt = d.get("T")
                p = d.get("p"); q = d.get("q"); m = d.get("m")
                if tt and p and q:
                    lat = recv_ms - tt
                    f.write(f"{recv_utc},{recv_ms:.3f},{tt},{tt},{lat:.3f},"
                            f"{p},{q},{m}\n")
                    msg += 1
                    if msg % FLUSH_EVERY == 0:
                        f.flush()
        except websockets.ConnectionClosed:
            print("[FUT-TRD] reconnect")
            continue


async def main():
    await asyncio.gather(spot_trades(), fut_bbo(), fut_trades())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")