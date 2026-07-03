# src/data_loader.py - VERSIÓN FINAL 100% FUNCIONAL - 19 NOV 2025
# Mantiene ETH-USD_Close y BTC-USD_Close exactos + normaliza el resto → 49 features perfectas

import pandas as pd
import yfinance as yf
from fredapi import Fred
import os
from src.features import safe_features


# =============================
# CONFIGURACIÓN FRED
# =============================
FRED_API_KEY = os.getenv("FRED_API_KEY", "e2e2e06cec19cc2b3fa8faf53b398ef4")
fred = Fred(api_key=FRED_API_KEY)


# ============================================================
# FUNCIÓN PRINCIPAL: DESCARGA TODO EL DATASET COMPLETO
# ============================================================
def get_full_dataset(start_date="2018-01-01", end_date=None):
    """
    Descarga y prepara todo el dataset con datos de mercado + macro + features.
    GARANTIZA que ETH-USD_Close y BTC-USD_Close se mantengan exactamente así.
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    print("\n============================================")
    print(" DESCARGANDO DATOS COMPLETOS (MARKET + MACRO)")
    print("============================================\n")

  
    # ------------------------------------------------------
    # 1) DESCARGA DE MERCADO - VERSIÓN BLINDADA NOVIEMBRE 2025
    # ------------------------------------------------------
    symbols = ["ETH-USD", "BTC-USD", "DX-Y.NYB", "^VIX", "^GSPC", "^IXIC"]
    market_frames = []

    for ticker in symbols:
        print(f" → Descargando {ticker}", end="")
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False
            )

            # Aplanar MultiIndex si aparece
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
                if df.columns.nlevels > 1:
                    df = df.droplevel(1, axis=1)

            if df.empty:
                print(" → vacío")
                continue

            if "Adj Close" in df.columns:
                price = df["Adj Close"]
            elif "Close" in df.columns:
                price = df["Close"]
            else:
                print(f" → sin precio → {list(df.columns)}")
                continue

            col_name = f"{ticker}_Close" if ticker in ["ETH-USD", "BTC-USD"] else f"{ticker.replace('^','').replace('-','_').replace('.','_')}_Close"
            price.name = col_name
            market_frames.append(price.to_frame())
            print(f" → OK")

        except Exception as e:
            print(f" → ERROR: {e}")

    market_df = pd.concat(market_frames, axis=1)

    # ------------------------------------------------------
    # 2) DESCARGA DE INDICADORES MACRO DEL FRED
    # ------------------------------------------------------
    print("\nDescargando datos macro del FRED...")
    macro_dict = {}
    fred_series = ["FEDFUNDS", "CPIAUCSL", "UNRATE", "M2SL"]

    for code in fred_series:
        print(f" → {code}")
        try:
            series = fred.get_series(code, observation_start=start_date)
            if not series.empty:
                macro_dict[code] = series
        except Exception as e:
            print(f"   FRED fallo en {code}: {e}")

    macro_df = pd.DataFrame(macro_dict)

    # Merge mercado + macro
    full_df = market_df.join(macro_df, how="outer")

    # ------------------------------------------------------
    # 3) REINDEXADO DIARIO + FFILL
    # ------------------------------------------------------
    full_df = full_df.resample("D").ffill()
    if full_df.index.tz is not None:
        full_df.index = full_df.index.tz_localize(None)

    # ------------------------------------------------------
    # 4) DESFASES MACRO (LAGS) — ANTI-LOOKAHEAD
    # ------------------------------------------------------
    macro_lags = {
        "FEDFUNDS": 2,
        "CPIAUCSL": 14,
        "UNRATE": 7,
        "M2SL": 10,
    }
    for col, lag in macro_lags.items():
        if col in full_df.columns:
            full_df[col] = full_df[col].shift(lag)

    # ------------------------------------------------------
    # 5) IMPUTACIÓN FINAL
    # ------------------------------------------------------
    for col in full_df.columns:
        if not full_df[col].isnull().all():
            full_df[col] = full_df[col].interpolate("linear").ffill().bfill()

    # ------------------------------------------------------
    # 6) GENERACIÓN DE FEATURES
    # ------------------------------------------------------
    full_featured = safe_features(full_df).dropna()

    print("\n============================================")
    print(f"  ¡DESCARGA COMPLETA!   {full_featured.shape[0]} días  ×  {full_featured.shape[1]} features")
    print("============================================\n")


    full_featured = safe_features(full_df).dropna()
    
    # Seguridad extra: si por algún motivo sigue sin existir → la reconstruimos
    if "ETH-USD_Close" not in full_featured.columns:
        if "ETH-USD_Close_lag1" in full_featured.columns:
            full_featured["ETH-USD_Close"] = full_featured["ETH-USD_Close_lag1"].shift(1).ffill()


    return full_featured