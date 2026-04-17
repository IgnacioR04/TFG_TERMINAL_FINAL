#!/usr/bin/env python3
"""
terminal_tfg.py - Generador de la terminal estatica
====================================================

Descarga datos de mercado (yfinance + alternative.me), calcula indicadores
tecnicos manualmente, estima un proxy de regimen de mercado y un semaforo
de volatilidad, serializa todo a JSON y lo inyecta en un template HTML
que se despliega en GitHub Pages.

La idea es que este script se ejecute via GitHub Actions cada X minutos
(cron mas bajo soportado: 5 min, en la practica 10 min), para que la
terminal siempre muestre datos razonablemente recientes.

Uso local:
    pip install yfinance pandas numpy requests
    python terminal_tfg.py

Salida: index.html en la misma carpeta, listo para commit + push.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import requests


# ============================================================
# CONFIG
# ============================================================

TICKERS_INDICES = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Dow Jones": "^DJI",
    "IBEX 35": "^IBEX",
    "EURO STOXX 50": "^STOXX50E",
}

TICKERS_US = {
    "Apple": "AAPL", "NVIDIA": "NVDA", "Microsoft": "MSFT",
    "Amazon": "AMZN", "Tesla": "TSLA", "Alphabet": "GOOGL", "Meta": "META",
}
TICKERS_EU = {
    "ASML": "ASML", "LVMH": "MC.PA", "SAP": "SAP",
    "Siemens": "SIE.DE", "TotalEnergies": "TTE.PA", "Novo Nordisk": "NVO",
}
TICKERS_ES = {
    "Inditex": "ITX.MC", "Santander": "SAN.MC", "BBVA": "BBVA.MC",
    "Iberdrola": "IBE.MC", "Telefonica": "TEF.MC", "Repsol": "REP.MC",
}

TICKERS_MACRO = {
    "dxy": "DX-Y.NYB",
    "oro": "GC=F",
    "bonos": "^TNX",
    "wti": "CL=F",
    "vix": "^VIX",
}


# ============================================================
# DESCARGA
# ============================================================

def dl(ticker, periodo, intervalo):
    """Descarga un ticker de yfinance. Si falla devuelve DF vacio."""
    try:
        df = yf.download(ticker, period=periodo, interval=intervalo, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        print(f"  [!] {ticker}: {e}")
        return pd.DataFrame()


def dl_fg():
    """Descarga el Fear & Greed Index de alternative.me (ultimos 365 dias)."""
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=365&format=json", timeout=15)
        data = r.json()["data"]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df["value"] = df["value"].astype(int)
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        print(f"  [!] F&G: {e}")
        return pd.DataFrame()


# ============================================================
# INDICADORES TECNICOS (calculo manual, sin pandas-ta)
# ============================================================

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    ef = ema(close, fast)
    es = ema(close, slow)
    line = ef - es
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

def atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def bbands(close, n=20, k=2):
    m = close.rolling(n).mean()
    s = close.rolling(n).std()
    return m + k * s, m, m - k * s


# ============================================================
# REGIMEN (proxy HMM light) + SEMAFORO VOLATILIDAD (proxy GARCH)
# ============================================================

def regimen_proxy(close, ventana=72):
    """
    Proxy simple del regimen HMM del TFG. Se clasifica cada vela en
    BULL / BEAR / SIDEWAYS segun el retorno acumulado en una ventana
    rolling y la pendiente normalizada.

    Nota: esto NO es el HMM del TFG (que queda privado). Es un proxy
    visual basado solo en dinamica de precio, sin exponer el modelo.
    """
    lr = np.log(close / close.shift(1))
    ret_ac = lr.rolling(ventana).sum()
    vol = lr.rolling(ventana).std()
    z = ret_ac / (vol * np.sqrt(ventana))
    reg = pd.Series(index=close.index, dtype=object)
    reg[z > 0.5] = "BULL"
    reg[z < -0.5] = "BEAR"
    reg[(z >= -0.5) & (z <= 0.5)] = "SIDEWAYS"
    return reg.fillna("SIDEWAYS")


def segmentar_regimen(reg, idx):
    """Convierte una serie de regimen en bloques [inicio, fin, tipo].

    Se consolidan bloques consecutivos del mismo tipo y se fusionan
    transiciones muy cortas (< 8 horas) con el bloque anterior, para
    evitar ruido visual y reducir el tamano del payload JSON.
    """
    if len(reg) == 0:
        return []
    bloques = []
    cur = reg.iloc[0]
    ini = idx[0]
    for i in range(1, len(reg)):
        if reg.iloc[i] != cur:
            bloques.append({
                "t0": int(ini.timestamp() * 1000),
                "t1": int(idx[i].timestamp() * 1000),
                "r": cur,
            })
            cur = reg.iloc[i]
            ini = idx[i]
    bloques.append({
        "t0": int(ini.timestamp() * 1000),
        "t1": int(idx[-1].timestamp() * 1000),
        "r": cur,
    })

    # Consolidamos: si un bloque dura menos de 8h y esta rodeado del
    # mismo regimen, lo fusionamos. Esto limpia mucho el ruido.
    min_dur = 8 * 3600 * 1000
    consolidado = [bloques[0]]
    for b in bloques[1:]:
        last = consolidado[-1]
        dur = b["t1"] - b["t0"]
        if dur < min_dur and last["r"] != b["r"]:
            # Fusion con anterior (extendemos el anterior)
            last["t1"] = b["t1"]
        elif last["r"] == b["r"]:
            last["t1"] = b["t1"]
        else:
            consolidado.append(b)
    return consolidado


def semaforo_vol(atr_pct_series):
    """Semaforo de volatilidad basado en percentil historico del ATR%."""
    if atr_pct_series is None or len(atr_pct_series.dropna()) < 50:
        return {"p": None, "val": None, "label": "N/A"}
    serie = atr_pct_series.dropna()
    actual = float(serie.iloc[-1])
    pct = float((serie < actual).mean() * 100)
    if pct <= 50:
        label = "BAJA"
    elif pct <= 80:
        label = "MEDIA"
    else:
        label = "ALTA"
    return {"p": round(pct, 1), "val": round(actual, 3), "label": label}


def ventanas_activas(reg, vol_pct_hist, vol_cur_pct):
    """
    Proxy de 'ventanas donde el modelo del TFG tendria mayor probabilidad':
    regimen definido (no transicion) + volatilidad por debajo del percentil 80.
    Se devuelve un array de bloques para pintar en la grafica como zonas
    activas sombreadas. NO expone el modelo ni sus predicciones.
    """
    if reg is None or vol_pct_hist is None:
        return []
    mask = pd.Series(False, index=reg.index)
    if len(vol_pct_hist) == len(reg):
        mask = vol_pct_hist <= 80
    bloques = []
    idx = reg.index
    activa = False
    ini = None
    for i in range(len(reg)):
        if mask.iloc[i] and reg.iloc[i] in ("BULL", "BEAR", "SIDEWAYS"):
            if not activa:
                activa = True
                ini = idx[i]
        else:
            if activa:
                activa = False
                bloques.append({"t0": int(ini.timestamp() * 1000), "t1": int(idx[i].timestamp() * 1000)})
    if activa:
        bloques.append({"t0": int(ini.timestamp() * 1000), "t1": int(idx[-1].timestamp() * 1000)})
    # Nos quedamos solo con los bloques de al menos 12h para que no quede ruidoso
    min_dur = 12 * 3600 * 1000
    return [b for b in bloques if b["t1"] - b["t0"] >= min_dur]


# ============================================================
# INGESTA COMPLETA
# ============================================================

def ingestar():
    D = {}
    print("=" * 54)
    print("TERMINAL TFG - Ingesta de datos")
    print("=" * 54)

    print("\n[1/7] BTC + ETH (horario, 1 ano)...")
    D["btc"] = dl("BTC-USD", "1y", "1h")
    D["eth"] = dl("ETH-USD", "1y", "1h")
    print(f"  BTC: {len(D['btc'])} velas | ETH: {len(D['eth'])} velas")

    print("[2/7] Indices (diario, 1 ano)...")
    D["indices"] = {n: dl(t, "1y", "1d") for n, t in TICKERS_INDICES.items()}

    print("[3/7] Empresas US/EU/ES (diario, 1 ano)...")
    D["us"] = {n: dl(t, "1y", "1d") for n, t in TICKERS_US.items()}
    D["eu"] = {n: dl(t, "1y", "1d") for n, t in TICKERS_EU.items()}
    D["es"] = {n: dl(t, "1y", "1d") for n, t in TICKERS_ES.items()}

    print("[4/7] Macro (diario, 1 ano)...")
    D["macro"] = {n: dl(t, "1y", "1d") for n, t in TICKERS_MACRO.items()}

    print("[5/7] Fear & Greed (1 ano)...")
    D["fg"] = dl_fg()

    print("[6/7] Indicadores tecnicos BTC...")
    bi = D["btc"].copy()
    if len(bi) > 50:
        bi["RSI"] = rsi(bi["Close"], 14)
        ml, ms, mh = macd(bi["Close"])
        bi["MACD"] = ml
        bi["MACD_sig"] = ms
        bi["MACD_hist"] = mh
        bi["ATR"] = atr(bi["High"], bi["Low"], bi["Close"], 14)
        bi["ATR_pct"] = bi["ATR"] / bi["Close"] * 100
        bb_u, bb_m, bb_l = bbands(bi["Close"], 20, 2)
        bi["BB_up"] = bb_u
        bi["BB_md"] = bb_m
        bi["BB_lo"] = bb_l
        bi["BB_width"] = (bb_u - bb_l) / bb_m
        vm = bi["Volume"].rolling(24).mean()
        vs = bi["Volume"].rolling(24).std()
        bi["Vol_zscore"] = (bi["Volume"] - vm) / vs
        bi["EMA21"] = ema(bi["Close"], 21)
        bi["EMA50"] = ema(bi["Close"], 50)
        bi["EMA200"] = ema(bi["Close"], 200)
    D["btc_ind"] = bi

    print("[7/7] Regimen y semaforo (proxy)...")
    if len(bi) > 100:
        reg = regimen_proxy(bi["Close"], ventana=72)
        D["reg_serie"] = reg
        D["reg_bloques"] = segmentar_regimen(reg, bi.index)
        vol_pct_hist = bi["ATR_pct"].expanding().rank(pct=True) * 100
        D["vol_pct_hist"] = vol_pct_hist
        D["sem"] = semaforo_vol(bi["ATR_pct"])
        D["ventanas_act"] = ventanas_activas(reg, vol_pct_hist, D["sem"]["p"])
        D["reg_actual"] = str(reg.iloc[-1])
    else:
        D["reg_serie"] = pd.Series(dtype=object)
        D["reg_bloques"] = []
        D["vol_pct_hist"] = pd.Series(dtype=float)
        D["sem"] = {"p": None, "val": None, "label": "N/A"}
        D["ventanas_act"] = []
        D["reg_actual"] = "N/A"

    # Correlacion BTC-ETH rolling 24h
    if len(D["btc"]) > 24 and len(D["eth"]) > 24:
        ci = D["btc"].index.intersection(D["eth"].index)
        cd = pd.DataFrame({
            "br": D["btc"].loc[ci, "Close"].pct_change(),
            "er": D["eth"].loc[ci, "Close"].pct_change(),
        })
        cd["corr"] = cd["br"].rolling(24).corr(cd["er"])
        D["corr"] = cd
    else:
        D["corr"] = pd.DataFrame()

    print("\nIngesta completada.\n")
    return D


# ============================================================
# UTILIDADES DE SERIALIZACION
# ============================================================

def ts_ms(idx):
    return [int(t.timestamp() * 1000) for t in idx]

def slist(s, d=2):
    if s is None:
        return []
    return [round(float(v), d) if pd.notna(v) else None for v in s]

def precio(df):
    return float(df["Close"].iloc[-1]) if len(df) > 0 else None

def cambio(df, n=1):
    if len(df) < n + 1:
        return None
    a, b = float(df["Close"].iloc[-1]), float(df["Close"].iloc[-(n + 1)])
    return ((a - b) / b) * 100 if b else None

def fp(v, d=2):
    if v is None:
        return "N/A"
    return f"${v:,.{d}f}" if abs(v) >= 1000 else f"${v:.{d}f}"

def fe(v, d=2):
    return f"\u20ac{v:,.{d}f}" if v else "N/A"

def pc(v):
    if v is None:
        return "N/A", ""
    return f"{'+' if v >= 0 else ''}{v:.2f}%", "up" if v >= 0 else "down"


def sparkline(df, n=30, d=2):
    """Devuelve una lista de cierres (ultimos n) para dibujar una mini-grafica."""
    if df is None or len(df) == 0:
        return []
    s = df["Close"].tail(n).tolist()
    return [round(float(v), d) if pd.notna(v) else None for v in s]


# ============================================================
# SERIALIZACION A JSON (lo que consume el HTML)
# ============================================================

def serializar(D):
    J = {}

    # BTC OHLC + EMAs
    # Estrategia de downsampling para reducir el tamano del HTML:
    #  - Conservamos los ultimos 3 meses en resolucion 1h (timeframes 7D/30D/90D)
    #  - El resto del ano se downsamplea a 4h (timeframes 6M/1Y)
    # Resultado: pasamos de 8760 puntos a ~3700, manteniendo fidelidad visual.
    b = D["btc"]
    bi = D["btc_ind"]
    if len(b) > 0:
        cutoff = b.index[-1] - pd.Timedelta(days=92)
        reciente = b[b.index >= cutoff]
        antiguo = b[b.index < cutoff]
        if len(antiguo) > 0:
            antiguo_ds = antiguo.resample("4h").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
            }).dropna()
            b_final = pd.concat([antiguo_ds, reciente])
        else:
            b_final = reciente

        J["btc"] = {
            "d": ts_ms(b_final.index),
            "o": slist(b_final["Open"]),
            "h": slist(b_final["High"]),
            "l": slist(b_final["Low"]),
            "c": slist(b_final["Close"]),
        }

        # EMAs y BB sobre el mismo indice
        if len(bi) > 0:
            bi_final = bi.reindex(b_final.index, method="nearest")
            J["btc_ema"] = {
                "ema21": slist(bi_final.get("EMA21", pd.Series())),
                "ema50": slist(bi_final.get("EMA50", pd.Series())),
                "ema200": slist(bi_final.get("EMA200", pd.Series())),
                "bb_up": slist(bi_final.get("BB_up", pd.Series())),
                "bb_lo": slist(bi_final.get("BB_lo", pd.Series())),
            }
        else:
            J["btc_ema"] = {"ema21": [], "ema50": [], "ema200": [], "bb_up": [], "bb_lo": []}
    else:
        J["btc"] = {"d": [], "o": [], "h": [], "l": [], "c": []}
        J["btc_ema"] = {"ema21": [], "ema50": [], "ema200": [], "bb_up": [], "bb_lo": []}

    # Regimen: bloques de color + ventanas activas
    J["regimen"] = D.get("reg_bloques", [])
    J["ventanas_act"] = D.get("ventanas_act", [])

    # ETH: mismo downsampling
    e = D["eth"]
    if len(e) > 0:
        cutoff = e.index[-1] - pd.Timedelta(days=92)
        reciente = e[e.index >= cutoff]
        antiguo = e[e.index < cutoff]
        if len(antiguo) > 0:
            antiguo_ds = antiguo["Close"].resample("4h").last().dropna()
            e_close = pd.concat([antiguo_ds, reciente["Close"]])
        else:
            e_close = reciente["Close"]
        J["eth"] = {"d": ts_ms(e_close.index), "v": slist(e_close)}
    else:
        J["eth"] = {"d": [], "v": []}

    # Correlacion: solo ultimos 6 meses (sobra)
    c = D.get("corr", pd.DataFrame())
    if len(c) > 0:
        c_tail = c.tail(24 * 180)
        J["corr"] = {"d": ts_ms(c_tail.index), "v": slist(c_tail["corr"], 3)}
    else:
        J["corr"] = {"d": [], "v": []}

    # Indices (diarios, sin downsampling)
    J["indices"] = {}
    for n, df in D["indices"].items():
        J["indices"][n] = {"d": ts_ms(df.index), "v": slist(df["Close"])} if len(df) > 0 else {"d": [], "v": []}

    # Acciones con sparkline (30d) para cards tipo Revolut
    for reg in ["us", "eu", "es"]:
        stk = []
        for n, df in D[reg].items():
            p = precio(df)
            eur = reg in ["eu", "es"]
            stk.append({
                "n": n,
                "tk": n,
                "reg": reg,
                "p": fe(p) if eur else fp(p),
                "p_raw": round(p, 2) if p else None,
                "d1": round(cambio(df, 1) or 0, 2),
                "d5": round(cambio(df, 5) or 0, 2),
                "d30": round(cambio(df, 22) or 0, 2),
                "spark": sparkline(df, 30),
                "line": {"d": ts_ms(df.index), "v": slist(df["Close"])} if len(df) > 0 else {"d": [], "v": []},
            })
        J[f"stk_{reg}"] = stk

    # Macro con sparklines (diarios, sin downsampling)
    J["macro"] = {}
    for k in ["dxy", "oro", "bonos", "wti", "vix"]:
        df = D["macro"].get(k, pd.DataFrame())
        J["macro"][k] = {
            "d": ts_ms(df.index) if len(df) > 0 else [],
            "v": slist(df["Close"]) if len(df) > 0 else [],
            "spark": sparkline(df, 30, 3),
        }

    # Indicadores tecnicos BTC: aplicamos el mismo downsampling que a BTC para alinear
    if len(bi) > 0:
        cutoff = bi.index[-1] - pd.Timedelta(days=92)
        reciente_bi = bi[bi.index >= cutoff]
        antiguo_bi = bi[bi.index < cutoff]

        def ds_series(col):
            if col not in bi.columns:
                return pd.Series(dtype=float)
            r = reciente_bi[col]
            if len(antiguo_bi) > 0:
                a = antiguo_bi[col].resample("4h").mean().dropna()
                return pd.concat([a, r])
            return r

        rsi_s = ds_series("RSI")
        atr_s = ds_series("ATR_pct")
        vz_s = ds_series("Vol_zscore")
        ml_s = ds_series("MACD")
        ms_s = ds_series("MACD_sig")
        mh_s = ds_series("MACD_hist")
        vph = D.get("vol_pct_hist", pd.Series(dtype=float))
        if len(vph) > 0:
            reciente_v = vph[vph.index >= cutoff]
            antiguo_v = vph[vph.index < cutoff]
            if len(antiguo_v) > 0:
                vph_ds = pd.concat([antiguo_v.resample("4h").mean().dropna(), reciente_v])
            else:
                vph_ds = reciente_v
        else:
            vph_ds = pd.Series(dtype=float)

        J["rsi"] = {"d": ts_ms(rsi_s.index), "v": slist(rsi_s, 1)}
        J["atr"] = {"d": ts_ms(atr_s.index), "v": slist(atr_s, 3)}
        J["vz"] = {"d": ts_ms(vz_s.index), "v": slist(vz_s, 2)}
        J["macd"] = {
            "d": ts_ms(ml_s.index),
            "ml": slist(ml_s),
            "ms": slist(ms_s),
            "mh": slist(mh_s),
        }
        J["vol_pct_hist"] = {"d": ts_ms(vph_ds.index), "v": slist(vph_ds, 1)}
    else:
        J["rsi"] = J["atr"] = J["vz"] = {"d": [], "v": []}
        J["macd"] = {"d": [], "ml": [], "ms": [], "mh": []}
        J["vol_pct_hist"] = {"d": [], "v": []}

    # Fear & Greed
    fg = D["fg"]
    if len(fg) > 0:
        J["fg"] = {
            "d": [int(t.timestamp() * 1000) for t in fg["timestamp"]],
            "v": fg["value"].tolist(),
            "cls": fg["value_classification"].tolist() if "value_classification" in fg.columns else [],
        }
    else:
        J["fg"] = {"d": [], "v": [], "cls": []}

    return J


# ============================================================
# KPIs (para header y cards)
# ============================================================

def kpis(D):
    K = {}
    btc, eth = D["btc"], D["eth"]

    K["btc_p"] = fp(precio(btc), 0)
    d1 = cambio(btc, 24)
    K["btc_c"], K["btc_cl"] = pc(d1)
    K["btc_hi"] = fp(float(btc["High"].tail(24).max()), 0) if len(btc) > 24 else "N/A"
    K["btc_lo"] = fp(float(btc["Low"].tail(24).min()), 0) if len(btc) > 24 else "N/A"
    K["btc_vol"] = f"${float(btc['Volume'].tail(24).sum()) / 1e9:.1f}B" if len(btc) > 24 else "N/A"
    K["eth_p"] = fp(precio(eth), 2)
    d1e = cambio(eth, 24)
    K["eth_c"], K["eth_cl"] = pc(d1e)

    # Macro
    for k, lbl, dec, dol in [("dxy", "DXY", 2, False), ("oro", "Oro", 0, True),
                              ("bonos", "Bonos", 2, False), ("wti", "WTI", 2, True),
                              ("vix", "VIX", 1, False)]:
        df = D["macro"].get(k, pd.DataFrame())
        p = precio(df)
        c = cambio(df)
        cs, cl = pc(c)
        if k == "bonos":
            pf = f"{p:.2f}%" if p else "N/A"
        elif dol:
            pf = fp(p, dec)
        else:
            pf = f"{p:.{dec}f}" if p else "N/A"
        K[f"m_{k}_p"], K[f"m_{k}_c"], K[f"m_{k}_cl"] = pf, cs, cl

    # Indicadores tecnicos
    bi = D["btc_ind"]
    if len(bi) > 0:
        rv = bi["RSI"].iloc[-1] if "RSI" in bi.columns and pd.notna(bi["RSI"].iloc[-1]) else None
        K["rsi"] = f"{rv:.1f}" if rv else "N/A"
        K["rsi_t"] = "Sobrecompra" if rv and rv > 70 else "Sobreventa" if rv and rv < 30 else "Neutral"
        av = bi["ATR_pct"].iloc[-1] if "ATR_pct" in bi.columns and pd.notna(bi["ATR_pct"].iloc[-1]) else None
        K["atr"] = f"{av:.2f}%" if av else "N/A"
        vv = bi["Vol_zscore"].iloc[-1] if "Vol_zscore" in bi.columns and pd.notna(bi["Vol_zscore"].iloc[-1]) else None
        K["vzv"] = f"{vv:.2f}" if vv else "N/A"
        mv = bi["MACD"].iloc[-1] if "MACD" in bi.columns and pd.notna(bi["MACD"].iloc[-1]) else None
        K["macd_v"] = f"{'+' if mv >= 0 else ''}{mv:.1f}" if mv is not None else "N/A"
        K["macd_cl"] = "up" if mv is not None and mv >= 0 else "down"
    else:
        K.update({"rsi": "N/A", "rsi_t": "", "atr": "N/A", "vzv": "N/A", "macd_v": "N/A", "macd_cl": ""})

    # Semaforo volatilidad
    sem = D.get("sem", {"p": None, "label": "N/A", "val": None})
    K["sem_label"] = sem["label"]
    K["sem_p"] = f"{sem['p']:.0f}" if sem["p"] is not None else "?"
    K["sem_val"] = f"{sem['val']:.2f}%" if sem["val"] is not None else "?"
    if sem["label"] == "BAJA":
        K["sem_col"] = "var(--neon)"
    elif sem["label"] == "MEDIA":
        K["sem_col"] = "var(--amber)"
    elif sem["label"] == "ALTA":
        K["sem_col"] = "var(--red)"
    else:
        K["sem_col"] = "var(--text3)"

    # Regimen actual
    K["reg_actual"] = D.get("reg_actual", "N/A")
    if K["reg_actual"] == "BULL":
        K["reg_col"] = "var(--neon)"
    elif K["reg_actual"] == "BEAR":
        K["reg_col"] = "var(--red)"
    elif K["reg_actual"] == "SIDEWAYS":
        K["reg_col"] = "var(--amber)"
    else:
        K["reg_col"] = "var(--text3)"

    # Ventanas activas (solo conteo, no exponemos mas)
    K["va_count"] = len(D.get("ventanas_act", []))
    # Porcentaje del ultimo mes que ha estado en ventana activa
    if D.get("ventanas_act"):
        now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
        mes_atras = now_ms - 30 * 24 * 3600 * 1000
        dur_activa = 0
        for v in D["ventanas_act"]:
            t0 = max(v["t0"], mes_atras)
            t1 = min(v["t1"], now_ms)
            if t1 > t0:
                dur_activa += t1 - t0
        K["va_pct_mes"] = f"{(dur_activa / (30 * 24 * 3600 * 1000)) * 100:.0f}%"
    else:
        K["va_pct_mes"] = "0%"

    # F&G
    fg = D["fg"]
    if len(fg) > 0:
        fv = int(fg["value"].iloc[-1])
        K["fg_v"] = str(fv)
        K["fg_t"] = fg["value_classification"].iloc[-1] if "value_classification" in fg.columns else ""
        K["fg_cl"] = "up" if fv > 50 else "down"
    else:
        K["fg_v"], K["fg_t"], K["fg_cl"] = "N/A", "", ""

    # Indices KPI (para tab RV)
    idx_html = ""
    idx_data = []
    for n in TICKERS_INDICES:
        df = D["indices"].get(n, pd.DataFrame())
        p = precio(df)
        c = cambio(df)
        cs, cl = pc(c)
        pf = f"{p:,.0f}" if p else "N/A"
        spark = sparkline(df, 30)
        idx_data.append({"n": n, "p": pf, "c": cs, "cl": cl, "spark": spark})
    K["idx_data"] = idx_data

    return K


# ============================================================
# HTML TEMPLATE
# ============================================================

def generar_html(D, template_path="template.html"):
    J = serializar(D)
    K = kpis(D)

    now = dt.datetime.now().strftime("%d/%m/%Y %H:%M")

    # Leemos el template base
    tpl = Path(template_path).read_text(encoding="utf-8")

    # Inyeccion de datos: serializamos todo lo dinamico en una unica variable JS
    payload = {
        "meta": {"generated_at": now, "now_ms": int(dt.datetime.now().timestamp() * 1000)},
        "kpi": K,
        "data": J,
        "tickers_indices": list(TICKERS_INDICES.keys()),
    }
    payload_js = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    # Inyeccion de KPIs basicos en el HTML estatico (para que se vean sin JS)
    # Clase CSS del badge de regimen segun estado actual
    reg_class = {"BULL": "bull", "BEAR": "bear", "SIDEWAYS": "side"}.get(K["reg_actual"], "side")

    reemplazos = {
        "__GEN_AT__": now,
        "__BTC_P__": K["btc_p"],
        "__BTC_C__": K["btc_c"],
        "__BTC_CL__": K["btc_cl"],
        "__ETH_P__": K["eth_p"],
        "__ETH_C__": K["eth_c"],
        "__ETH_CL__": K["eth_cl"],
        "__FG_V__": K["fg_v"],
        "__FG_T__": K["fg_t"],
        "__FG_CL__": K["fg_cl"],
        "__REG__": K["reg_actual"],
        "__REG_COL__": K["reg_col"],
        "__REG_CLASS__": reg_class,
        "__SEM__": K["sem_label"],
        "__SEM_COL__": K["sem_col"],
        "__SEM_P__": K["sem_p"],
        "__SEM_VAL__": K["sem_val"],
        "__VA_PCT__": K["va_pct_mes"],
        "__PAYLOAD_JSON__": payload_js,
    }

    html = tpl
    for k, v in reemplazos.items():
        html = html.replace(k, str(v))

    return html


# ============================================================
# MAIN
# ============================================================

def main():
    D = ingestar()
    html = generar_html(D, template_path="template.html")
    out = Path("index.html")
    out.write_text(html, encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"\nindex.html generado ({kb:.0f} KB)")
    print(f"Ruta: {out.resolve()}")


if __name__ == "__main__":
    main()
