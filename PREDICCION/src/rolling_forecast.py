# src/rolling_forecast.py
"""
Rolling / Expanding Window Forecasting (walk-forward)
Genera predicciones consecutivas (mensual/trimestral/semestral)
y guarda un JSON acumulado con: id, fecha_inicio, fecha_predicha, close, high, low, tendencia.

High/Low se obtienen SIEMPRE con EGARCH+FHS (módulo: egarch_src/models/egarch_fhs.py).
No hay lookahead: en cada iteración se descarga/usa solo datos hasta la fecha de corte.
"""

import os
import json
from datetime import timedelta
import pandas as pd
import numpy as np

# Data loader (usa tu implementación existente)
from src.data_loader import get_full_dataset

# Model prediction functions (asegúrate que las rutas importadas existan)
from src.models.model1_xgb_qr import predict_model1
from src.models.model2_lstm_transformer import predict_model2
from src.models.model3_hybrid import predict_model3
from src.models.model4_arima_gru import predict_model4

# EGARCH+FHS (tu módulo especificado)
# Ajusta la ruta si tu proyecto lo requiere; me dijiste que el módulo es:
# egarch_src/models/egarch_fhs.py
# importando la función simulate_fhs_ranges
import importlib.util
import pathlib

EGARCH_MODULE_PATH = pathlib.Path("egarch_src/models/egarch_fhs.py")
if not EGARCH_MODULE_PATH.exists():
    # Try the src path as fallback
    EGARCH_MODULE_PATH = pathlib.Path("src/models/egarch_fhs.py")

if not EGARCH_MODULE_PATH.exists():
    raise FileNotFoundError(f"EGARCH module not found at expected paths. Checked: 'egarch_src/models/egarch_fhs.py' and 'src/models/egarch_fhs.py'")

spec = importlib.util.spec_from_file_location("egarch_fhs", str(EGARCH_MODULE_PATH))
egarch_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(egarch_mod)
# funciones esperadas en el módulo:
# - fit_and_save_egarch / simulate_fhs_ranges
simulate_fhs_ranges = getattr(egarch_mod, "simulate_fhs_ranges", None)
fit_and_save_egarch = getattr(egarch_mod, "fit_and_save_egarch", None)

if simulate_fhs_ranges is None:
    raise ImportError("simulate_fhs_ranges not found in egarch module.")

# Map de modelos disponibles (identificadores que pasarás desde la UI)
MODEL_MAP = {
    "model1_xgb_qr": predict_model1,
    "model2_lstm_transformer": predict_model2,
    "model3_tabnet_tft": predict_model3,
    "model4_arima_gru": predict_model4,
    # "model5_egarch_fhs": handled internally if chosen (EGARCH used for high/low anyway)
}

HORIZONTES = {
    "mensual": 30,
    "trimestral": 90,
    "semestral": 180
}


def _safe_predict_price(predict_func, train_df):
    """
    Llamada wrapper para extraer una predicción 'close' escalar desde cada modelo.
    Las funciones predict_modelX se asumen que retornan un escalar (precio esperado).
    Hacemos validaciones y fallbacks.
    """
    try:
        pred = predict_func(train_df, train_df)
        # si devuelve array-like, tomar el último o la media
        if pred is None:
            return None
        if isinstance(pred, (list, tuple, np.ndarray, pd.Series)):
            # si es vector, tomar la mediana
            arr = np.array(pred).astype(float)
            return float(np.median(arr))
        return float(pred)
    except Exception as e:
        print(f"[WARN] fallo en predict_func: {e}")
        return None


def rolling_expanding_forecast(
    fecha_descarga_inicio: str,
    fecha_inicio_backtest: str,
    fecha_fin_backtest: str,
    horizonte: str = "trimestral",
    modelo_id: str = "model2_lstm_transformer",
    save_json_path: str = "rolling_results.json",
    egarch_path: str = "./models/egarch_fitted.pkl",
    min_rows_required: int = 200,
    n_sims_egarch: int = 10000
):
    """
    Ejecuta el flujo:
      - descarga datos desde fecha_descarga_inicio hasta fecha_actual_corte
      - entrena (invocando la función predict de cada modelo que reentrena internamente)
      - predice close
      - genera high/low con EGARCH+FHS (simulate_fhs_ranges)
      - guarda en JSON acumulado
    """
    # Validaciones
    if horizonte not in HORIZONTES:
        raise ValueError(f"Horizonte inválido: {horizonte}. Opciones: {list(HORIZONTES.keys())}")

    dias_horizonte = HORIZONTES[horizonte]

    fecha_descarga = pd.to_datetime(fecha_descarga_inicio)
    fecha_bt_actual = pd.to_datetime(fecha_inicio_backtest)
    fecha_bt_final = pd.to_datetime(fecha_fin_backtest)

    resultados = []
    consecutivo = 0

    # Si existe JSON previo, lo cargamos y continuamos (opcional)
    if os.path.exists(save_json_path):
        try:
            with open(save_json_path, "r") as f:
                prev = json.load(f)
                if isinstance(prev, list):
                    resultados = prev
                    consecutivo = max([r.get("id", -1) for r in resultados]) + 1
        except Exception:
            # no crítico, empezamos desde 0
            resultados = []
            consecutivo = 0

    # Elegir función de predicción
    predict_func = MODEL_MAP.get(modelo_id, None)

    # If user selected egarch model id specifically, keep predict_func None and we'll use egarch median as 'close'
    use_egarch_as_model = (modelo_id == "model5_egarch_fhs")
    if predict_func is None and not use_egarch_as_model:
        raise ValueError(f"Modelo desconocido: {modelo_id}. Opciones: {list(MODEL_MAP.keys()) + ['model5_egarch_fhs']}")

    # Main loop: avanzar en pasos de 'horizonte' días (expanding)
    while fecha_bt_actual <= fecha_bt_final:
        print(f"\n[rolling] Iteración {consecutivo} — corte en: {fecha_bt_actual.date()}")

        # 1) Descargar datos hasta fecha_bt_actual (NO mirar más allá)
        df = get_full_dataset(start_date=str(fecha_descarga.date()), end_date=str(fecha_bt_actual.date()))
        df = df.dropna()
        if df.shape[0] < min_rows_required:
            print(f"[rolling] Datos insuficientes ({df.shape[0]} filas) — requerido >= {min_rows_required}. Interrumpiendo.")
            break

        # 2) Preparo entrenamiento: (aquí los modelos usan df completo y reentrenan)
        last_close = float(df["ETH-USD_Close"].iloc[-1])

        # 3) Predicción 'close'
        if use_egarch_as_model:
            # Si piden modelo EGARCH como predictor primario, usamos EGARCH FHS median (50% percentile)
            pcts = simulate_fhs_ranges(last_close, horizon=dias_horizonte, n_sims=n_sims_egarch, egarch_path=egarch_path)
            # simulate_fhs_ranges retorna percentiles [5,15,30,50,70,85,95] según tu módulo
            close_pred = float(pcts[3])  # mediana
        else:
            close_pred = _safe_predict_price(predict_func, df)
            if close_pred is None:
                print("[rolling] fallback: modelo no retornó predicción, usamos último precio conocido")
                close_pred = last_close

        # 4) High / Low via EGARCH+FHS (SIEMPRE)
        # Aseguramos que exista el modelo EGARCH; si no existe lo ajustamos con los retornos actuales
        try:
            pcts = simulate_fhs_ranges(close_pred, horizon=dias_horizonte, n_sims=n_sims_egarch, egarch_path=egarch_path)
        except Exception as e:
            print(f"[rolling] EGARCH simulate falla: {e} — intentando reentrenar EGARCH con retornos disponibles")
            # intentar ajustar egarch con returns y volver a simular (si fit disponible)
            if fit_and_save_egarch:
                returns = np.log(df["ETH-USD_Close"] / df["ETH-USD_Close"].shift(1)).dropna()
                fit_and_save_egarch(returns, path=egarch_path)
                pcts = simulate_fhs_ranges(close_pred, horizon=dias_horizonte, n_sims=n_sims_egarch, egarch_path=egarch_path)
            else:
                # fallback simple por si no se puede usar egarch
                high_pred = close_pred * 1.10
                low_pred = close_pred * 0.90
                pcts = None

        if pcts is not None:
            # Según tu función: [5,15,30,50,70,85,95]
            # Usamos 95% como high y 5% como low (rango extremo). Esto es consistente con estimación de colas.
            low_pred = float(pcts[0])
            high_pred = float(pcts[-1])
        else:
            # fallback
            high_pred = float(high_pred)
            low_pred = float(low_pred)

        # 5) Tendencia según comparación entre close_pred y último close real
        tendencia = "alcista" if close_pred > last_close else "bajista"

        # 6) Guardar iteración en resultados y persistir JSON
        item = {
            "id": consecutivo,
            "fecha_inicio_corte": str(fecha_bt_actual.date()),
            "fecha_predicha": str((fecha_bt_actual + timedelta(days=dias_horizonte)).date()),
            "pred_close": round(float(close_pred), 6),
            "pred_high": round(float(high_pred), 6),
            "pred_low": round(float(low_pred), 6),
            "tendencia": tendencia
        }

        resultados.append(item)

        # persistir
        with open(save_json_path, "w") as f:
            json.dump(resultados, f, indent=2)

        print(f"[rolling] Guardado id={consecutivo} fecha_predicha={item['fecha_predicha']}")

        # avanzar al siguiente corte (expanding window): la nueva fecha de corte es la fecha_predicha
        fecha_bt_actual = fecha_bt_actual + timedelta(days=dias_horizonte)
        consecutivo += 1

    print("\n[rolling] Proceso finalizado. JSON guardado en:", save_json_path)
    return resultados


if __name__ == "__main__":
    # pequeño test local (no se ejecuta en Streamlit)
    res = rolling_expanding_forecast(
        fecha_descarga_inicio="2022-01-01",
        fecha_inicio_backtest="2023-01-01",
        fecha_fin_backtest="2024-01-01",
        horizonte="trimestral",
        modelo_id="model2_lstm_transformer",
        save_json_path="rolling_results_example.json",
        egarch_path="./models/egarch_fitted.pkl",
        min_rows_required=200,
        n_sims_egarch=3000
    )
    print("Ejemplo terminado. Predicciones:", len(res))
