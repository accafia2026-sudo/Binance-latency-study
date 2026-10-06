"""
Generate publication-quality figures for the latency paper.

Reads gzipped CSVs from the final snapshot directory and produces:
    fig1_latency_cdf.pdf       — CDF of latency per stream (log-x)
    fig2_hourly_drift.pdf      — Hourly p50/p99 across capture
    fig3_event_c_timeseries.pdf — Event C: futures vs spot, 60s window
    fig4_epsilon_vs_lambda.pdf — ε_t vs λ_panic (quadratic fit)
    fig5_tail_counts.pdf       — Bar chart of >30ms, >100ms, >500ms counts
    fig6_exceedance.pdf        — Survival function P(L > x) per stream
    fig7_burst_boxplots.pdf    — Box plots calm vs burst, per stream

Usage:
    python make_figures.py /path/to/data_dir /path/to/output_dir
"""
import gzip
import io
import os
import sys
import zlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import LogLocator, FuncFormatter

# ---------- Style ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Colour palette (colorblind-safe)
C_FUT   = "#d62728"  # red
C_SPOT  = "#2ca02c"  # green
C_BBO   = "#1f77b4"  # blue
C_CALM  = "#7f7f7f"  # gray
C_BURST = "#ff7f0e"  # orange


DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    r"D:\latency-research\data\final_snapshot"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else \
    r"D:\latency-research\paper\figures"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------- Reader ----------
def read_truncated_gz(path, **kwargs):
    """Tolerant gzip reader for actively-written files."""
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


# ---------- Load ----------
print("Loading data...")
fut = read_truncated_gz(
    os.path.join(DATA_DIR, "btc_fut_trades.csv.gz"),
    usecols=["recv_utc", "latency_ms", "price", "qty", "is_buyer_maker"])
fut["recv_utc"] = pd.to_datetime(fut["recv_utc"], format="ISO8601", utc=True)
fut = fut[fut["price"] > 1000].reset_index(drop=True)

spot = read_truncated_gz(
    os.path.join(DATA_DIR, "btc_spot_trades.csv.gz"),
    usecols=["recv_utc", "latency_ms", "price", "qty"])
spot["recv_utc"] = pd.to_datetime(spot["recv_utc"], format="ISO8601", utc=True)
spot = spot[spot["price"] > 1000].reset_index(drop=True)

bbo = read_truncated_gz(
    os.path.join(DATA_DIR, "btc_fut_bbo.csv.gz"),
    usecols=["recv_utc", "latency_ms"])
bbo["recv_utc"] = pd.to_datetime(bbo["recv_utc"], format="ISO8601", utc=True)

print(f"  futures trades: {len(fut):,}")
print(f"  spot trades:    {len(spot):,}")
print(f"  futures BBO:    {len(bbo):,}")


# ============================================================
# FIG 1 — CDF of latency per stream (log-x)
# ============================================================
def fig1_latency_cdf():
    fig, ax = plt.subplots(figsize=(8, 5))

    for df, name, color in [
        (bbo, "Futures BBO", C_BBO),
        (fut, "Futures trades", C_FUT),
        (spot, "Spot trades", C_SPOT),
    ]:
        lat = df["latency_ms"].dropna().to_numpy()
        lat = lat[(lat > 0) & (lat < 10000)]
        lat.sort()
        cdf = np.arange(1, len(lat) + 1) / len(lat)
        # Downsample for plotting speed
        step = max(1, len(lat) // 20000)
        ax.plot(lat[::step], cdf[::step], label=name, color=color, lw=1.8)

    ax.set_xscale("log")
    ax.set_xlabel("Latency $L$ (ms)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("Latency CDF by stream")
    ax.axvline(30, color="black", ls=":", lw=1, alpha=0.6)
    ax.axvline(100, color="black", ls=":", lw=1, alpha=0.6)
    ax.text(30, 0.02, " 30 ms", fontsize=9, color="black")
    ax.text(100, 0.02, " 100 ms", fontsize=9, color="black")
    ax.legend(loc="lower right", frameon=True)
    ax.set_ylim(0, 1.02)
    ax.grid(True, which="both", alpha=0.25, ls="--")
    fig.savefig(os.path.join(OUT_DIR, "fig1_latency_cdf.pdf"))
    plt.close(fig)
    print("  fig1 done")


# ============================================================
# FIG 2 — Hourly latency drift
# ============================================================
def fig2_hourly_drift():
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for df, name, color in [
        (fut, "Futures trades", C_FUT),
        (spot, "Spot trades", C_SPOT),
        (bbo, "Futures BBO", C_BBO),
    ]:
        s = df.set_index("recv_utc")["latency_ms"]
        hourly = s.resample("1h").agg(
            p50="median",
            p99=lambda x: x.quantile(0.99),
        )
        axes[0].plot(hourly.index, hourly["p50"], "o-",
                     label=name, color=color, lw=1.6, ms=3)
        axes[1].plot(hourly.index, hourly["p99"], "o-",
                     label=name, color=color, lw=1.6, ms=3)

    axes[0].set_ylabel("Hourly p50 latency (ms)")
    axes[1].set_ylabel("Hourly p99 latency (ms)")
    axes[0].set_title("Hourly latency drift over 27-hour capture")
    for ax in axes:
        ax.legend(loc="best", frameon=True)
        ax.grid(True, alpha=0.3, ls="--")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[1].xaxis.set_major_locator(mdates.HourLocator(interval=3))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_hourly_drift.pdf"))
    plt.close(fig)
    print("  fig2 done")


# ============================================================
# FIG 3 — Event C: futures vs spot 60-second window
# ============================================================
def fig3_event_c_timeseries():
    t_c = pd.Timestamp("2026-10-05 15:32:00", tz="UTC")
    t0, t1 = t_c - pd.Timedelta("30s"), t_c + pd.Timedelta("30s")

    fw = fut[(fut["recv_utc"] >= t0) & (fut["recv_utc"] <= t1)].copy()
    sw = spot[(spot["recv_utc"] >= t0) & (spot["recv_utc"] <= t1)].copy()

    # Bin into 1-second buckets for readability
    fw["sec"] = fw["recv_utc"].dt.floor("1s")
    sw["sec"] = sw["recv_utc"].dt.floor("1s")
    f_binned = fw.groupby("sec")["latency_ms"].median()
    s_binned = sw.groupby("sec")["latency_ms"].median()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    # Top: latency
    axes[0].plot(f_binned.index, f_binned.values, "o-",
                 label="Futures trade feed", color=C_FUT, lw=1.8, ms=4)
    axes[0].plot(s_binned.index, s_binned.values, "s-",
                 label="Spot trade feed", color=C_SPOT, lw=1.8, ms=4)
    axes[0].axvline(t_c, color="black", ls="--", alpha=0.5,
                    label="Burst center")
    axes[0].set_ylabel("Median latency (ms)")
    axes[0].set_title("Event C — 15:32 UTC, 5 Oct 2026 (±30 s window)")
    axes[0].legend(loc="upper left", frameon=True)
    axes[0].grid(True, alpha=0.3, ls="--")

    # Bottom: price
    fp = fw.groupby("sec")["price"].last()
    axes[1].plot(fp.index, fp.values, "-", color=C_FUT, lw=1.5)
    axes[1].axvline(t_c, color="black", ls="--", alpha=0.5)
    axes[1].set_ylabel("BTCUSDT price (\\$)")
    axes[1].set_xlabel("Time (UTC)")
    axes[1].grid(True, alpha=0.3, ls="--")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_event_c_timeseries.pdf"))
    plt.close(fig)
    print("  fig3 done")


# ============================================================
# FIG 4 — epsilon vs lambda_panic (quadratic relationship)
# ============================================================
def fig4_epsilon_vs_lambda():
    # Data from Table 2
    events = pd.DataFrame([
        {"label": "Calm",   "lam": 10.9,  "L": 4.07,  "eps": 0.04},
        {"label": "A",      "lam": 157.7, "L": 80.98, "eps": 12.77},
        {"label": "B",      "lam": 82.5,  "L": 92.04, "eps": 7.59},
        {"label": "C",      "lam": 343.5, "L": 47.00, "eps": 16.14},
    ])
    events["eps_predicted"] = events["lam"] * events["L"] / 1000.0

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(events["lam"], events["eps"], s=100, zorder=5,
               color=C_BURST, edgecolor="black", linewidth=1.2)
    for _, row in events.iterrows():
        ax.annotate(row["label"], (row["lam"], row["eps"]),
                    textcoords="offset points", xytext=(8, 6),
                    fontsize=10, fontweight="bold")

    # Quadratic reference curve
    x = np.linspace(5, 400, 100)
    # Calibrate so curve passes near event C
    k = events.loc[events["label"] == "C", "eps"].iloc[0] / \
        (events.loc[events["label"] == "C", "lam"].iloc[0] ** 2)
    ax.plot(x, k * x**2, "--", color="gray", lw=1.5,
            label=r"Quadratic reference $\varepsilon \propto \lambda^2$")

    # Linear reference for comparison
    k_lin = events.loc[events["label"] == "C", "eps"].iloc[0] / \
            events.loc[events["label"] == "C", "lam"].iloc[0]
    ax.plot(x, k_lin * x, ":", color="black", lw=1.2,
            label=r"Linear reference $\varepsilon \propto \lambda$")

    ax.set_xlabel(r"Fill rate $\lambda_{\mathrm{panic}}$ (fills/s)")
    ax.set_ylabel(r"Inventory observability error $\varepsilon_t$ (fills)")
    ax.set_title(r"Super-linear growth of $\varepsilon_t$ in burst intensity")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3, ls="--")
    fig.savefig(os.path.join(OUT_DIR, "fig4_epsilon_vs_lambda.pdf"))
    plt.close(fig)
    print("  fig4 done")


# ============================================================
# FIG 5 — Tail count bars
# ============================================================
def fig5_tail_counts():
    counts = pd.DataFrame({
        "Stream": ["Futures trades", "Spot trades", "Futures BBO"],
        ">30ms":  [(fut["latency_ms"] > 30).sum(),
                    (spot["latency_ms"] > 30).sum(),
                    (bbo["latency_ms"] > 30).sum()],
        ">100ms": [(fut["latency_ms"] > 100).sum(),
                    (spot["latency_ms"] > 100).sum(),
                    (bbo["latency_ms"] > 100).sum()],
        ">500ms": [(fut["latency_ms"] > 500).sum(),
                    (spot["latency_ms"] > 500).sum(),
                    (bbo["latency_ms"] > 500).sum()],
    })
    # Normalize to percent of rows
    totals = {"Futures trades": len(fut), "Spot trades": len(spot),
              "Futures BBO": len(bbo)}
    for col in [">30ms", ">100ms", ">500ms"]:
        counts[col] = counts.apply(
            lambda r: 100 * r[col] / totals[r["Stream"]], axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(counts))
    width = 0.25
    cols = [">30ms", ">100ms", ">500ms"]
    colors = ["#f4a582", "#d6604d", "#b2182b"]

    for i, (col, c) in enumerate(zip(cols, colors)):
        vals = counts[col].values
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=col, color=c, edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() * 1.05,
                    f"{v:.2f}%" if v < 1 else f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(counts["Stream"])
    ax.set_ylabel("Fraction of messages exceeding threshold (%)")
    ax.set_title("Latency tail exceedance by stream")
    ax.set_yscale("log")
    ax.legend(title="Threshold", frameon=True)
    ax.grid(True, axis="y", alpha=0.3, ls="--")
    fig.savefig(os.path.join(OUT_DIR, "fig5_tail_counts.pdf"))
    plt.close(fig)
    print("  fig5 done")


# ============================================================
# FIG 6 — Survival function P(L > x)
# ============================================================
def fig6_exceedance():
    fig, ax = plt.subplots(figsize=(8, 5))

    x_grid = np.logspace(0, 3.5, 200)
    for df, name, color in [
        (fut, "Futures trades", C_FUT),
        (spot, "Spot trades", C_SPOT),
        (bbo, "Futures BBO", C_BBO),
    ]:
        lat = df["latency_ms"].dropna().to_numpy()
        lat = lat[(lat > 0) & (lat < 10000)]
        surv = np.array([(lat > x).mean() for x in x_grid])
        ax.plot(x_grid, surv, label=name, color=color, lw=1.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Latency threshold $x$ (ms)")
    ax.set_ylabel(r"$P(L > x)$")
    ax.set_title("Latency survival function by stream")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, which="both", alpha=0.25, ls="--")
    ax.set_xlim(1, 3000)
    ax.set_ylim(1e-6, 1.5)
    fig.savefig(os.path.join(OUT_DIR, "fig6_exceedance.pdf"))
    plt.close(fig)
    print("  fig6 done")


# ============================================================
# FIG 7 — Box plots: calm vs burst, per stream
# ============================================================
def fig7_burst_boxplots():
    calm_t0 = pd.Timestamp("2026-10-05 10:00", tz="UTC")
    calm_t1 = pd.Timestamp("2026-10-05 10:01", tz="UTC")
    burst_t0 = pd.Timestamp("2026-10-05 15:32", tz="UTC")
    burst_t1 = pd.Timestamp("2026-10-05 15:33", tz="UTC")

    def slice_lat(df, t0, t1):
        m = (df["recv_utc"] >= t0) & (df["recv_utc"] < t1)
        return df.loc[m, "latency_ms"].dropna().to_numpy()

    data = [
        slice_lat(fut, calm_t0, calm_t1),
        slice_lat(fut, burst_t0, burst_t1),
        slice_lat(spot, calm_t0, calm_t1),
        slice_lat(spot, burst_t0, burst_t1),
        slice_lat(bbo, calm_t0, calm_t1),
        slice_lat(bbo, burst_t0, burst_t1),
    ]
    labels = [
        "Futures\ncalm", "Futures\nburst",
        "Spot\ncalm", "Spot\nburst",
        "BBO\ncalm", "BBO\nburst",
    ]
    colors = [C_CALM, C_FUT, C_CALM, C_SPOT, C_CALM, C_BBO]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, labels=labels, showfliers=False,
                    patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.5)

    # Overlay jittered points
    for i, d in enumerate(data):
        if len(d) == 0:
            continue
        step = max(1, len(d) // 500)
        x_jit = np.random.normal(i + 1, 0.06, len(d[::step]))
        ax.scatter(x_jit, d[::step], s=2, alpha=0.25, color="black",
                   edgecolors="none")

    ax.set_yscale("log")
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_title("Latency distribution: calm window (10:00) vs burst window (15:32)")
    ax.grid(True, axis="y", alpha=0.3, ls="--")
    plt.setp(ax.get_xticklabels(), fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig7_burst_boxplots.pdf"))
    plt.close(fig)
    print("  fig7 done")


# ---------- Run all ----------
print(f"\nWriting figures to {OUT_DIR}")
fig1_latency_cdf()
fig2_hourly_drift()
fig3_event_c_timeseries()
fig4_epsilon_vs_lambda()
fig5_tail_counts()
fig6_exceedance()
fig7_burst_boxplots()
print("\nDone. Generated 7 PDF figures.")