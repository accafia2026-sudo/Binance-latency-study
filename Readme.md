binance-latency-study/
├── README.md
├── requirements.txt
├── scripts/
│   ├── probe_rest.py              # REST round-trip probe
│   ├── probe_ws.py                # Main multi-stream WS recorder
│   └── analyze.py                 # Analysis + event detection
└── data/
    └── (populated by the recorders)