#!/usr/bin/env python3
"""LG V6.9.20: K16/B4 research backtester for XAUUSD M30.
Faithful K16 neighbor-selection concept; feature normalization and some broker
execution assumptions are approximations, not a bit-exact Pine port.
Requires: pandas, numpy. Runs: python backtest.py
"""
import csv
from pathlib import Path
import numpy as np
import pandas as pd

CAPITAL_EUR = 500.0
LOT = 0.01
OUNCES_PER_LOT = 100.0
EURUSD = 1.10
LEVERAGE = 20.0
SPREAD_USD = 0.50
SLIPPAGE_USD = 0.05
FEE_EUR = 0.0
ATR_PERIOD = 14
SL_ATR = 2.5
TP_ATR = 4.0
MAX_HOLD_BARS = 48
HISTORY = 350
SAMPLE = 4
NEIGHBORS = 16
MAX_DD_PCT = 30.0
STOP_OUT_PCT = 50.0
START = "2024-01-01"
END = "2025-01-01"
OUT = Path("results")

def find_csv():
    candidates = [
        Path("data/XAUUSD_M30.csv"), Path("data/data/XAUUSD_M30.csv"),
        Path("XAUUSD_M30.csv"), Path("XAUUSD_M30_2024.csv"),
        Path("data/XAUUSD_M30_2024.csv")
    ]
    for p in candidates:
        if p.is_file():
            print("CSV:", p)
            return p
    raise FileNotFoundError("XAUUSD CSV fehlt. Erwartet: " +
                            ", ".join(map(str, candidates)))

def load_data(path):
    # Handles Dukascopy-style files, conventional CSV and tab/semicolon separators.
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip().lower().replace("<", "").replace(">", "") for c in df.columns]
    rename = {"datetime":"time", "timestamp":"time", "date":"time",
              "gmt time":"time", "local time":"time"}
    df = df.rename(columns=rename)
    if "time" not in df.columns:
        raise ValueError(f"Keine Zeitspalte gefunden: {list(df.columns)}")
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"Spalte {col} fehlt: {list(df.columns)}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    raw = df["time"]
    if pd.api.types.is_numeric_dtype(raw):
        unit = "ms" if pd.to_numeric(raw).median() > 1e11 else "s"
        df["time"] = pd.to_datetime(raw, unit=unit, utc=True, errors="coerce")
    else:
        df["time"] = pd.to_datetime(raw, utc=True, errors="coerce", dayfirst=False)
    df = df.dropna(subset=["time","open","high","low","close"])
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if df.empty:
        raise ValueError("CSV enthält keine gültigen Kerzen")
    return df

def rma(s, n):
    return s.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

def rsi(close, n):
    delta = close.diff()
    gain = rma(delta.clip(lower=0), n)
    loss = rma((-delta).clip(lower=0), n)
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)

def cci(high, low, close, n):
    tp = (high + low + close)/3
    ma = tp.rolling(n).mean()
    dev = tp.rolling(n).apply(lambda x: np.mean(np.abs(x-x.mean())), raw=True)
    return (tp-ma)/(0.015*dev.replace(0,np.nan))

def adx(high, low, close, n):
    up = high.diff()
    down = -low.diff()
    plus = pd.Series(np.where((up>down)&(up>0),up,0.0), index=close.index)
    minus = pd.Series(np.where((down>up)&(down>0),down,0.0), index=close.index)
    tr = pd.concat([(high-low), (high-close.shift()).abs(),
                    (low-close.shift()).abs()],axis=1).max(axis=1)
    atr = rma(tr,n)
    pdi = 100*rma(plus,n)/atr.replace(0,np.nan)
    mdi = 100*rma(minus,n)/atr.replace(0,np.nan)
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return rma(dx,n), atr

def wavetrend(high,low,close,n1=10,n2=11):
    ap = (high+low+close)/3
    esa = ap.ewm(span=n1,adjust=False).mean()
    d = (ap-esa).abs().ewm(span=n1,adjust=False).mean()
    ci = (ap-esa)/(0.015*d.replace(0,np.nan))
    wt = ci.ewm(span=n2,adjust=False).mean()
    return wt

def normalized(s, length=100):
    # Approximation of jdehorty/MLExtensions/2 normalization (library not bundled).
    mn = s.rolling(length,min_periods=20).min()
    mx = s.rolling(length,min_periods=20).max()
    return (100*(s-mn)/(mx-mn).replace(0,np.nan)).clip(0,100)

def indicators(df):
    h,l,c = df.high,df.low,df.close
    ad, atr = adx(h,l,c,20)
    _, atr14 = adx(h,l,c,ATR_PERIOD)
    # Feature order matches Pine defaults: RSI14, WT10/11, CCI20, ADX20, RSI9.
    features = np.column_stack([
        rsi(c,14).to_numpy(),
        normalized(wavetrend(h,l,c,10,11)).to_numpy(),
        normalized(cci(h,l,c,20)).to_numpy(),
        ad.to_numpy(),
        rsi(c,9).to_numpy()
    ])
    return features, atr14.to_numpy()

def k16_signals(df, features):
    n = len(df)
    close = df.close.to_numpy()
    scores = np.zeros(n,dtype=np.int8)
    # Label for historical bar j is price movement from j to j+4.
    # Candidate j <= current i-4 => no future data leakage.
    for i in range(HISTORY+6,n):
        f = features[i]
        if not np.isfinite(f).all():
            continue
        hs = np.arange(4,HISTORY+4)
        hs = hs[hs % SAMPLE == 0]
        js = i-hs
        valid = np.isfinite(features[js]).all(axis=1)
        js = js[valid]
        if len(js)<NEIGHBORS:
            continue
        distances = np.log1p(np.abs(features[js]-f)).sum(axis=1)
        # stable sort: earlier candidate wins ties, like Pine insertion order
        nearest = np.argsort(distances,kind="stable")[:NEIGHBORS]
        chosen = js[nearest]
        labels = np.sign(close[chosen+4]-close[chosen])
        scores[i] = np.sign(labels.sum()).astype(np.int8)
    return scores

def export_signal_audit(df, scores):
    """Export exact K16 transition rules, approximate feature inputs."""
    state = 0
    records = []
    for i in range(len(df)):
        t = df.time.iloc[i]
        if not (pd.Timestamp(START, tz="UTC") <= t < pd.Timestamp(END, tz="UTC")):
            continue
        score = int(scores[i])
        signal = ""
        if score == 1 and state != 1:
            signal = "BUY"
        elif score == -1 and state != -1:
            signal = "SELL"
        if score != 0:
            state = score
        records.append({"time_utc": t.isoformat(), "score_k16": score,
                        "signal_k16": signal, "state_k16": state})
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(OUT / "signals_python.csv", index=False)
    print("Signal-Audit:", OUT / "signals_python.csv")
    print("K16-Signalwechsel:", sum(bool(x["signal_k16"]) for x in records))


def simulate(df,scores,atr):
    OUT.mkdir(exist_ok=True)
    rows = []
    cash = CAPITAL_EUR
    peak = cash
    maxdd = 0.0
    halted = False
    position = None
    pending = 0
    state = 0
    rejected = 0
    stopouts = 0
    units = LOT*OUNCES_PER_LOT
    sidecost = SPREAD_USD/2+SLIPPAGE_USD
    dates = df.time.to_numpy()
    op,hi,lo,cl = (df[x].to_numpy() for x in ("open","high","low","close"))
    start = pd.Timestamp(START,tz="UTC")
    end = pd.Timestamp(END,tz="UTC")

    def exit_trade(i, price, reason):
        nonlocal cash, position, stopouts
        p = position
        exit_px = price-sidecost if p["side"]==1 else price+sidecost
        pnl = p["side"]*(exit_px-p["entry"])*units/EURUSD-FEE_EUR
        cash += pnl
        rows.append(dict(entry_time=str(df.time.iloc[p["bar"]]),
                         exit_time=str(df.time.iloc[i]), side="BUY" if p["side"]==1 else "SELL",
                         entry=round(p["entry"],3),exit=round(exit_px,3),
                         pnl_eur=round(pnl,2),balance_eur=round(cash,2),reason=reason))
        if reason=="STOP_OUT": stopouts+=1
        position=None

    for i in range(len(df)):
        t = df.time.iloc[i]
        in_window = start<=t<end
        if not in_window:
            if t>=end: break
            continue
        # At bar open: execute order placed after PREVIOUS closed candle.
        if pending and position is None and not halted:
            margin = op[i]*units/(LEVERAGE*EURUSD)
            if cash >= margin and np.isfinite(atr[i-1]) and atr[i-1]>0:
                side = pending
                entry = op[i]+side*sidecost
                position = dict(side=side,entry=entry,bar=i,
                                sl=entry-side*SL_ATR*atr[i-1],
                                tp=entry+side*TP_ATR*atr[i-1],margin=margin)
            else:
                rejected+=1
        pending=0

        if position is not None:
            p=position
            side=p["side"]
            # Conservative when SL and TP both occur in one candle: SL first.
            stop_hit = lo[i]<=p["sl"] if side==1 else hi[i]>=p["sl"]
            target_hit = hi[i]>=p["tp"] if side==1 else lo[i]<=p["tp"]
            if stop_hit:
                # Gap through stop fills at open if worse.
                fill=min(op[i],p["sl"]) if side==1 else max(op[i],p["sl"])
                exit_trade(i,fill,"SL")
            elif target_hit:
                fill=p["tp"]
                exit_trade(i,fill,"TP")
            elif i-p["bar"]>=MAX_HOLD_BARS:
                exit_trade(i,cl[i],"TIME")
            else:
                adverse = lo[i] if side==1 else hi[i]
                equity_worst = cash+side*(adverse-p["entry"])*units/EURUSD
                if equity_worst <= p["margin"]*STOP_OUT_PCT/100:
                    exit_trade(i,adverse,"STOP_OUT")

        equity = cash
        if position:
            equity += position["side"]*(cl[i]-position["entry"])*units/EURUSD
        peak=max(peak,equity)
        dd=100*(peak-equity)/peak if peak>0 else 100
        maxdd=max(maxdd,dd)
        if dd>=MAX_DD_PCT or equity<=0:
            halted=True
            if position:
                exit_trade(i,cl[i],"DRAWDOWN_HALT")

        # K16 score transition only, evaluated at completed candle close.
        if scores[i]!=0 and scores[i]!=state:
            state=int(scores[i])
            if position is None and not halted and i+1<len(df):
                pending=state
            elif position is not None and position["side"]!=state:
                # Opposite signal exit at close; new entry next candle.
                exit_trade(i,cl[i],"REVERSE")
                if not halted and i+1<len(df): pending=state

    if position:
        last = min(i,len(df)-1)
        exit_trade(last,cl[last],"END")
    pd.DataFrame(rows,columns=["entry_time","exit_time","side","entry","exit",
                                 "pnl_eur","balance_eur","reason"]).to_csv(
                                     OUT/"trades.csv",index=False)
    wins=sum(r["pnl_eur"]>0 for r in rows)
    losses=sum(r["pnl_eur"]<0 for r in rows)
    n=len(rows)
    print("\n=== LG V6.9.20 | K16/B4 | XAUUSD M30 ===")
    print("Zeitraum:",START,"bis",END,"(Ende exklusiv)")
    print("Trades:",n,"Gewinne:",wins,"Verluste:",losses)
    print("Winrate:",f"{wins/n*100:.2f}%" if n else "n/a")
    print("Startkapital:",f"{CAPITAL_EUR:.2f} EUR")
    print("Endkapital:",f"{cash:.2f} EUR")
    print("Netto:",f"{cash-CAPITAL_EUR:+.2f} EUR")
    print("Max. Drawdown:",f"{maxdd:.2f}%")
    print("Abgelehnte Entries:",rejected,"Stop-outs:",stopouts,
          "Trading angehalten:",halted)
    print("Trade-Log:",OUT/"trades.csv")

if __name__=="__main__":
    df=load_data(find_csv())
    print("Kerzen:",len(df),"von",df.time.iloc[0],"bis",df.time.iloc[-1])
    features,atr=indicators(df)
    scores=k16_signals(df,features)
    export_signal_audit(df,scores)
    simulate(df,scores,atr)
