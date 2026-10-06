"""
Analysis pipeline for the Binance latency study.

Loads gzipped CSVs produced by probe_ws.py, filters to valid price
ranges, and produces:
    - Table 1: latency distribution per stream
    - Table 2: burst events with epsilon_t
    - Hourly drift diagnostic

Handles truncated gzip files (may lack end-of-stream marker if the
recorder was still writing when the file was copied).

Usage:
    python analyze.py /path/to/data_dir
"""
import gzip
import io
import os
import sys
import zlib
import numpy as np
import pandas as pd

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/latency-research/data"
)


def read_truncated_gz(path, **kwargs):
    """
    Read a gzip file that may lack a valid end-of-stream marker.
    Returns a DataFrame with whatever decompressed successfully.
    """
    decompressed = bytearray()
    with open(path, "rb") as f:
        d = zlib.decompressobj(wbits=31)
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            try:
                decompressed.extend(d.decompress(chunk))
            except zlib.error:
                break
            if d.eof:
                break
    try:
        decompressed.extend(d.flush())
    except zlib.error:
        pass

    text = decompressed.decode("utf-8", errors="ignore")
    if text and not text.endswith("\n"):
        text = text[:text.rfind("\n") + 1]
    return pd.read_csv(io.StringIO(text), **kwargs)


def stream_stats(path, lat_col, name):
    """Print per-stream latency distribution (Table 1)."""
    df = read_truncated_gz(path, usecols=["recv_utc", "latency_ms"])
    df["recv_utc"] = pd.to_datetime(df["recv_utc"], format="ISO8601", utc=True)
    lat = df["latency_ms"]
    print(f"\n{name}")
    print(f"  rows    = {len(df):,}")
    print(f"  range   = {df['recv_utc'].iloc[0]} -> {df['recv_utc'].iloc[-1]}")
    print(f"  p50     = {lat.median():.2f} ms")
    print(f"  p99     = {lat.quantile(0.99):.2f} ms")
    print(f"  p999    = {lat.quantile(0.999):.2f} ms")
    print(f"  max     = {lat.max():.2f} ms")
    print(f"  >30ms   = {(lat > 30).sum():,}  ({(lat > 30).mean()*100:.3f}%)")
    print(f"  >100ms  = {(lat > 100).sum():,}  ({(lat > 100).mean()*100:.4f}%)")
    print(f"  >500ms  = {(lat > 500).sum():,}")
    return df


def hourly_drift(df, name):
    """Print hourly p50/p99 to demonstrate absence of instance throttling."""
    df = df.set_index("recv_utc").sort_index()
    h = df["latency_ms"].resample("1h").agg(
        p50="median",
        p99=lambda x: x.quantile(0.99),
        n="count",
    )
    print(f"\nHourly drift — {name}")
    print(h.round(2).to_string())


def detect_events(trades_df, z_thr=3.0, bps_thr=3.0, count_mult=3.0):
    """
    Detect order-flow bursts using three simultaneous conditions:
        |z| > z_thr AND |ret_bps| > bps_thr AND count_ratio > count_mult
    """
    df = trades_df.copy()
    df["recv_utc"] = pd.to_datetime(df["recv_utc"], format="ISO8601", utc=True)
    df = df[df["price"] > 1000].set_index("recv_utc").sort_index()

    px_1s = df["price"].resample("1s").last().dropna()
    cnt_1s = df["price"].resample("1s").count()

    ret = np.log(px_1s).diff().dropna()
    r_bps = ret * 1e4
    base = ret.rolling(60, min_periods=10).std().clip(lower=1e-6)
    z = ret / base
    cmed = cnt_1s.rolling(60, min_periods=10).median()
    cratio = cnt_1s.reindex(z.index).fillna(1.0) / cmed.reindex(z.index).fillna(1.0)

    mask = (z.abs() > z_thr) & (r_bps.abs() > bps_thr) & (cratio > count_mult)
    events = pd.DataFrame({
        "price": px_1s.reindex(z.index),
        "ret_bps": r_bps,
        "z": z,
        "count_ratio": cratio,
    })[mask]

    print(f"\nDetected {len(events)} burst events")
    if len(events):
        print(events.to_string())
    return events


def compute_epsilon(fut_df, spot_df, bbo_df, events,
                    window_s=60, avg_qty=0.01):
    """
    Compute the Avellaneda-Stoikov inventory observability error
    epsilon_t = lambda_panic * L_trade for each event.

    Returns a DataFrame with columns:
        event_ts, lambda (fills/s), L_fut (ms), L_spot (ms), L_bbo (ms),
        epsilon (fills), epsilon_btc, epsilon_usd
    """
    for df in (fut_df, spot_df, bbo_df):
        df["recv_utc"] = pd.to_datetime(
            df["recv_utc"], format="ISO8601", utc=True
        )

    rows = []
    for ts in events.index:
        t0 = ts - pd.Timedelta(seconds=window_s / 2)
        t1 = ts + pd.Timedelta(seconds=window_s / 2)

        fw = fut_df[(fut_df["recv_utc"] >= t0) & (fut_df["recv_utc"] <= t1)]
        sw = spot_df[(spot_df["recv_utc"] >= t0) & (spot_df["recv_utc"] <= t1)]
        bw = bbo_df[(bbo_df["recv_utc"] >= t0) & (bbo_df["recv_utc"] <= t1)]

        if len(fw) < 10:
            continue

        lam = len(fw) / window_s
        L_fut = fw["latency_ms"].median()
        L_spot = sw["latency_ms"].median() if len(sw) else np.nan
        L_bbo = bw["latency_ms"].median() if len(bw) else np.nan

        eps_fills = lam * (L_fut / 1000.0)
        eps_btc = eps_fills * avg_qty
        px = fw["price"].median()
        eps_usd = eps_btc * px

        rows.append({
            "event_ts": ts.isoformat(),
            "lambda_fills_per_s": round(lam, 2),
            "L_fut_ms": round(L_fut, 2),
            "L_spot_ms": round(L_spot, 2) if not np.isnan(L_spot) else None,
            "L_bbo_ms": round(L_bbo, 2) if not np.isnan(L_bbo) else None,
            "epsilon_fills": round(eps_fills, 3),
            "epsilon_btc": round(eps_btc, 5),
            "epsilon_usd": round(eps_usd, 2),
        })

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("LATENCY STUDY — ANALYSIS")
    print("=" * 60)

    fut = stream_stats(
        os.path.join(DATA_DIR, "btc_fut_trades.csv.gz"),
        lat_col=4, name="FUTURES TRADES"
    )
    spot = stream_stats(
        os.path.join(DATA_DIR, "btc_spot_trades.csv.gz"),
        lat_col=3, name="SPOT TRADES"
    )
    bbo = stream_stats(
        os.path.join(DATA_DIR, "btc_fut_bbo.csv.gz"),
        lat_col=4, name="FUTURES BBO"
    )

    hourly_drift(fut, "FUTURES TRADES")

    events = detect_events(fut)
    if len(events):
        eps = compute_epsilon(fut, spot, bbo, events, window_s=60)
        print("\n=== epsilon_t per event (1-minute windows) ===")
        print(eps.to_string(index=False))
        eps.to_csv(os.path.join(DATA_DIR, "epsilon_table.csv"), index=False)


if __name__ == "__main__":
    main()