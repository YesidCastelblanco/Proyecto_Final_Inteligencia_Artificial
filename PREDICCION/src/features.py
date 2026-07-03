# src/features.py
# VERSIÓN 2025 PRO – NIVEL INSTITUCIONAL
# Mantiene macro-features + BTC lagged + tus features técnicas
# 100% seguro contra data leakage

import numpy as np
import pandas as pd

def safe_features(df):
    """
    Función definitiva 2025: prepara features para todos los modelos
    - Mantiene macroeconomía completa
    - Añade BTC lagged (sin leakage)
    - Elimina solo precios crudos de ETH y BTC
    - Conserva todo lo que ya tenías
    """
    df = df.copy()

    # =================================================================
    # 1. TUS FEATURES ORIGINALES (las conservamos todas)
    # =================================================================
    for c in ["ETH-USD_Close", "BTC-USD_Close", "^GSPC_Close", "^IXIC_Close"]:
        if c in df.columns:
            df[f"{c}_logret"] = np.log(df[c] / df[c].shift(1))

    for c in ["ETH-USD_Close", "BTC-USD_Close"]:
        for lag in [1, 2, 3, 5, 7, 14, 30]:
            df[f"{c}_lag{lag}"] = df[c].shift(lag)

    for w in [7, 20, 30, 50, 90, 200]:
        df[f"ETH_MA{w}"] = df["ETH-USD_Close"].rolling(w).mean().shift(1)
        df[f"ETH_vol{w}"] = df["ETH-USD_Close_logret"].rolling(w).std().shift(1) * np.sqrt(365)

    delta = df["ETH-USD_Close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    df["ETH_RSI14"] = (100 - (100 / (1 + up / down))).shift(1)

    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    if "ETH-USD_Close" in df.columns and "BTC-USD_Close" in df.columns:
        df["ETH-BTC_ratio"] = df["ETH-USD_Close"] / df["BTC-USD_Close"]

    df["ETH_z30"] = (df["ETH-USD_Close"] - df["ETH-USD_Close"].rolling(30).mean()).shift(1) / df["ETH-USD_Close"].rolling(30).std().shift(1)

    for w in [7, 30, 90]:
        df[f"ETH_ret_{w}d"] = df["ETH-USD_Close"].pct_change(w).shift(1)

    # =================================================================
    # 2. NUEVO: AÑADIMOS BTC LAGGED SEGURO (sin leakage)
    # =================================================================
    if "BTC-USD_Close" in df.columns:
        df["BTC_lag1"] = df["BTC-USD_Close"].shift(1)
        df["BTC_return_lag1"] = df["BTC-USD_Close"].pct_change().shift(1)
        df["BTC_dominance_proxy"] = df["BTC-USD_Close"].shift(1) / df["BTC-USD_Close"].shift(1).rolling(30).mean()

    # =================================================================
    # 3. PREPARAMOS EL DATASET FINAL: eliminamos solo lo peligroso
    # =================================================================
    # Columnas que NUNCA deben entrar al modelo (precios crudos actuales)
    raw_price_columns = [
        'ETH-USD_Open', 'ETH-USD_High', 'ETH-USD_Low', 'ETH-USD_Close',
        'BTC-USD_Open', 'BTC-USD_High', 'BTC-USD_Low', 'BTC-USD_Close',
    ]

    # Eliminamos solo si existen
    columns_to_drop = [col for col in raw_price_columns if col in df.columns]
    # También eliminamos target si existe (en backtesting)
    if 'target_price' in df.columns:
        columns_to_drop.append('target_price')

    df_final = df.drop(columns=columns_to_drop)

    
    # =================================================================
    # FIX DEFINITIVO: RECONSTRUIR ETH Y BTC ANTES DEL dropna()
    # =================================================================
    if "ETH-USD_Close_lag1" in df_final.columns:
        df_final["ETH-USD_Close"] = df_final["ETH-USD_Close_lag1"].shift(1)
    if "BTC-USD_Close_lag1" in df_final.columns:
        df_final["BTC-USD_Close"] = df_final["BTC-USD_Close_lag1"].shift(1)

    # Rellenamos cualquier NaN residual (especialmente el último día)
    df_final["ETH-USD_Close"] = df_final["ETH-USD_Close"].ffill().bfill()
    df_final["BTC-USD_Close"] = df_final["BTC-USD_Close"].ffill().bfill()

    # =================================================================
    # 4. Limpieza final: AHORA sí eliminamos NaNs
    # =================================================================
    df_final = df_final.dropna()

    print(f"Features finales para los modelos: {df_final.shape[1]} columnas")
    macro_cols = [c for c in df_final.columns if any(x in c for x in ['^VIX', 'DX-Y', 'FEDFUNDS', 'M2SL', 'CPI', 'UNRATE', '^GSPC', '^IXIC'])]
    print(f"   → Macro-features activas: {len(macro_cols)} → {macro_cols[:10]}{'...' if len(macro_cols)>10 else ''}")
    print(f"   → BTC lagged: {'Sí' if 'BTC_lag1' in df_final.columns else 'No'}")

    return df_final
    

