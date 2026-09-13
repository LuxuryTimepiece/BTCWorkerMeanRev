#!/usr/bin/env python3
"""
BTC Mean-Reversion Worker
=========================
Philosophy: Buy dips inside a FRIENDLY daily regime.
- Only active when daily regime is FRIENDLY (strict)
- Entry: price near/below daily EMA20 or pulling back toward rising daily EMA50,
         4H RSI oversold-ish (< 40) and starting to turn up, MACD hist not strongly negative and improving
- Tight stops (ATR x 1.4), smaller risk (0.75%)
- Goal: harvest the pullback itself, not the trend continuation

Independent worker. Own state, own equity, own /status.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from flask import Flask, jsonify
import json, time, threading, os, warnings
from datetime import datetime, timezone
from pathlib import Path
warnings.filterwarnings("ignore")

SYMBOL = "BTC-USD"
INTERVAL = "4h"
EMA_FAST = 12
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
RSI_LEN = 14
RSI_OVERSOLD = 40
ATR_LEN, ATR_STOP_MULT = 14, 1.4
TRAIL_ACTIVATE_R, TRAIL_ATR_MULT = 0.6, 1.3
RISK_PCT = 0.0075
FEE_RT = 0.0004
INITIAL_EQUITY = 10_000.0
CHECK_INTERVAL = 300
DAILY_EMA_LEN = 50
DAILY_EMA_FAST = 20
MAX_DRAWDOWN_PCT = 0.06
MAX_CONSECUTIVE_LOSSES = 3
MIN_ATR_PCT = 0.004
MAX_BARS = 36  # ~6 days — mean-reversion should resolve faster

STATE_FILE = Path("paper_state_meanrev.json")

def default_state():
    return {
        "worker": "meanrev",
        "philosophy": "mean-reversion (dip buying in friendly regime)",
        "equity": INITIAL_EQUITY,
        "peak_equity": INITIAL_EQUITY,
        "position": 0,
        "entry_price": 0.0,
        "stop_price": 0.0,
        "orig_stop_pct": 0.0,
        "bars_held": 0,
        "last_bar_time": None,
        "trades": [],
        "equity_curve": [{"time": datetime.now(timezone.utc).isoformat(), "equity": INITIAL_EQUITY}],
        "last_update": datetime.now(timezone.utc).isoformat(),
        "status": "starting",
        "regime": "unknown",
        "risk_state": "unknown",
        "agent_decision": "unknown",
        "last_price": None,
    }

def load_state():
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            for k, v in default_state().items():
                if k not in state:
                    state[k] = v
            return state
        except Exception:
            pass
    return default_state()

def save_state(state):
    state["last_update"] = datetime.now(timezone.utc).isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def _norm(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df

def fetch_4h():
    df = yf.download(SYMBOL, period="60d", interval=INTERVAL, progress=False)
    if df.empty:
        return None
    df = _norm(df.dropna())
    df["ema12"] = EMAIndicator(df["close"], EMA_FAST).ema_indicator()
    macd = MACD(df["close"], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
    df["macd_hist"] = macd.macd_diff()
    df["macd_hist_prev"] = df["macd_hist"].shift(1)
    df["rsi"] = RSIIndicator(df["close"], RSI_LEN).rsi()
    df["rsi_prev"] = df["rsi"].shift(1)
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], ATR_LEN).average_true_range()
    return df.dropna()

def fetch_daily():
    df = yf.download(SYMBOL, period="1y", interval="1d", progress=False)
    if df.empty:
        return None
    df = _norm(df.dropna())
    df["ema50"] = EMAIndicator(df["close"], DAILY_EMA_LEN).ema_indicator()
    df["ema20"] = EMAIndicator(df["close"], DAILY_EMA_FAST).ema_indicator()
    return df.dropna()

def evaluate_regime():
    df = fetch_daily()
    if df is None or len(df) < DAILY_EMA_LEN + 5:
        return "unknown", {}
    last = df.iloc[-1]
    prev = df.iloc[-6] if len(df) >= 6 else df.iloc[0]
    price = float(last["close"])
    ema50 = float(last["ema50"])
    ema20 = float(last["ema20"])
    slope_up = ema50 >= float(prev["ema50"])
    above50 = price > ema50
    # Mean-reversion only wants clear FRIENDLY
    if above50 and slope_up:
        regime = "FRIENDLY"
    elif not above50:
        regime = "HOSTILE"
    else:
        regime = "MIXED"
    return regime, {
        "daily_close": round(price, 2),
        "daily_ema50": round(ema50, 2),
        "daily_ema20": round(ema20, 2),
        "ema50_slope_up": slope_up,
        "price_above_ema50": above50,
        "distance_to_ema20_pct": round((price - ema20) / ema20 * 100, 2),
    }

def evaluate_risk(state):
    equity = float(state.get("equity", INITIAL_EQUITY))
    peak = float(state.get("peak_equity", equity))
    if equity > peak:
        peak = equity
        state["peak_equity"] = peak
    drawdown = (peak - equity) / peak if peak > 0 else 0.0
    consecutive = 0
    for t in reversed(state.get("trades", [])):
        if t.get("pnl_pct", 0) < 0:
            consecutive += 1
        else:
            break
    risk_off = (drawdown >= MAX_DRAWDOWN_PCT) or (consecutive >= MAX_CONSECUTIVE_LOSSES)
    return ("RISK-OFF" if risk_off else "RISK-ON",
            {"equity": round(equity, 2), "peak_equity": round(peak, 2),
             "drawdown_pct": round(drawdown * 100, 2), "consecutive_losses": consecutive})

def get_signal(row, regime_detail):
    """Dip buy: RSI was oversold and turning up, hist improving, price near daily support."""
    rsi = float(row["rsi"])
    rsi_prev = float(row["rsi_prev"])
    hist = float(row["macd_hist"])
    hist_prev = float(row["macd_hist_prev"])
    # RSI oversold zone and starting to rise
    rsi_turn = (rsi_prev <= RSI_OVERSOLD) and (rsi > rsi_prev)
    # Or RSI still low but rising
    rsi_low_rising = (rsi < 45) and (rsi > rsi_prev)
    hist_improving = hist > hist_prev
    # Prefer price not too extended above daily EMA20 (still a dip)
    dist = regime_detail.get("distance_to_ema20_pct", 0)
    near_support = dist < 3.0  # within 3% of daily EMA20 or below
    return bool((rsi_turn or rsi_low_rising) and hist_improving and near_support)

def run_once(state):
    df = fetch_4h()
    if df is None or len(df) < 5:
        state["status"] = "data_error"
        return state
    latest = df.iloc[-1]
    bar_time = str(latest.name)
    price = float(latest["close"])
    atr = float(latest["atr"])
    atr_pct = atr / price if price > 0 else 0

    regime, regime_detail = evaluate_regime()
    risk_state, risk_detail = evaluate_risk(state)
    state["regime"] = regime
    state["risk_state"] = risk_state
    state["regime_detail"] = regime_detail
    state["risk_detail"] = risk_detail
    # Mean-reversion is strict: only FRIENDLY
    allow = (regime == "FRIENDLY") and (risk_state == "RISK-ON")
    state["agent_decision"] = "CONTINUE" if allow else "PAUSE"

    if state.get("last_bar_time") == bar_time:
        state["status"] = "waiting_for_new_bar"
        state["last_price"] = price
        return state

    if state["position"] == 1:
        state["bars_held"] = state.get("bars_held", 0) + 1
        entry, stop = state["entry_price"], state["stop_price"]
        r_mult = (price - entry) / (entry - stop) if entry > stop else 0
        if r_mult >= TRAIL_ACTIVATE_R:
            new_stop = price - atr * TRAIL_ATR_MULT
            if new_stop > stop:
                state["stop_price"] = new_stop
        exit_now, reason = False, ""
        if price <= state["stop_price"]:
            exit_now, reason = True, "stop"
        elif float(latest["rsi"]) >= 55 and float(latest["macd_hist"]) > 0:
            exit_now, reason = True, "target_zone"  # mean-rev takes profit into strength
        elif state["bars_held"] >= MAX_BARS:
            exit_now, reason = True, "time"
        if exit_now:
            raw = (price - entry) / entry
            pnl = raw - FEE_RT
            risk_d = state["equity"] * RISK_PCT
            size = risk_d / state["orig_stop_pct"] if state.get("orig_stop_pct", 0) > 1e-8 else 0
            dollar = size * pnl
            state["equity"] += dollar
            if state["equity"] > state.get("peak_equity", state["equity"]):
                state["peak_equity"] = state["equity"]
            state.setdefault("trades", []).append({
                "time": bar_time, "side": "long",
                "entry": round(entry, 2), "exit": round(price, 2),
                "pnl_pct": round(pnl * 100, 3), "dollar": round(dollar, 2),
                "reason": reason, "bars": state["bars_held"],
                "equity_after": round(state["equity"], 2),
            })
            state["position"] = 0
            state["bars_held"] = 0

    weekend = latest.name.weekday() >= 5 if hasattr(latest.name, "weekday") else False
    atr_ok = atr_pct >= MIN_ATR_PCT
    if state["position"] == 0 and allow and atr_ok and not weekend:
        if get_signal(latest, regime_detail) and atr > 0:
            stop_dist = max(atr * ATR_STOP_MULT, price * MIN_ATR_PCT * 1.2)
            state["position"] = 1
            state["entry_price"] = price
            state["stop_price"] = price - stop_dist
            state["orig_stop_pct"] = stop_dist / price
            state["bars_held"] = 0

    state["last_bar_time"] = bar_time
    state.setdefault("equity_curve", []).append({
        "time": datetime.now(timezone.utc).isoformat(), "equity": state["equity"]
    })
    if len(state["equity_curve"]) > 3000:
        state["equity_curve"] = state["equity_curve"][-2000:]
    state["status"] = "ok"
    state["last_price"] = price
    return state

def worker_loop():
    print("Mean-reversion worker loop started", flush=True)
    while True:
        try:
            state = load_state()
            state = run_once(state)
            save_state(state)
            pos = "LONG" if state["position"] == 1 else "FLAT"
            print(f"[meanrev {datetime.now().strftime('%H:%M:%S')}] {pos} | Eq ${state['equity']:.2f} | "
                  f"{state.get('regime')} / {state.get('risk_state')} / {state.get('agent_decision')}", flush=True)
        except Exception as e:
            print(f"Meanrev worker error: {e}", flush=True)
        time.sleep(CHECK_INTERVAL)

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"service": "BTC Mean-Reversion Worker", "endpoints": {"/status": "state", "/health": "health"}})

@app.route("/health")
def health():
    return jsonify({"status": "alive", "worker": "meanrev", "time": datetime.now(timezone.utc).isoformat()})

@app.route("/status")
def status():
    return jsonify(load_state())

_started = False
_lock = threading.Lock()
def start_once():
    global _started
    with _lock:
        if not _started:
            threading.Thread(target=worker_loop, daemon=True).start()
            _started = True
start_once()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8082))
    app.run(host="0.0.0.0", port=port)
