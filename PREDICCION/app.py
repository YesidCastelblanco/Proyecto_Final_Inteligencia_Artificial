# app.py - VERSIÓN FINAL PROFESIONAL 2025 - ARQUITECTURA QUANT DEFINITIVA
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os

from src.data_loader import get_full_dataset
from src.models.model1_xgb_qr import predict_model1
from src.models.model2_lstm_transformer import predict_model2, finetune_model2
from src.models.model3_hybrid import predict_model3, finetune_hybrid
from src.models.model4_arima_gru import predict_model4, finetune_gru
from src.models.volatility import fit_egarch_high_quality, simulate_fhs_egarch_v2, load_joblib
from src.backtesting import run_full_backtest

# ========================================
# CONFIG + TEMA CYBERPUNK
# ========================================
# ========================================
# CONFIG + TEMA CYBERPUNK ULTRA PREMIUM
# ========================================
st.set_page_config(
    page_title="ETH GridBot AI 2025 - Predicciones", 
    page_icon="🟢", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# CSS ÉPICO MEJORADO
st.markdown("""
<style>
    .main > div {max-width: 95%; padding-top: 2rem;}
    .stApp {background: linear-gradient(135deg, #0a0e17, #1a0033);}
    
    h1, h2, h3 {font-family: 'Orbitron', sans-serif !important; color: #00c27a !important; text-shadow: 0 0 40px #00ff9d88;}
    
    /* Tabs como botones cyberpunk - MÁS GRANDES */
    .stTabs [data-baseweb="tab-list"] {gap: 20px; justify-content: center; flex-wrap: wrap;}
    .stTabs [data-baseweb="tab"] {
        height: 90px !important; width: 400px !important; border-radius: 20px !important; font-size: 28px !important; font-weight: bold !important;
        background: linear-gradient(145deg, #1a1f2e, #16213e) !important; border: 3px solid #00ff9d44 !important;
        color: #00d1ff !important; transition: all 0.4s !important;
    }
    .stTabs [data-baseweb="tab"]:hover {border: 3px solid #00ff9d; transform: translateY(-8px); box-shadow: 0 15px 35px #00ff9d44;}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(145deg, #00ff9d22, #00d1ff22); border: 3px solid #00ff9d; color: white !important;
    }

    /* Botones épicos - MÁS GRANDES */
    .stButton > button {
        background: linear-gradient(90deg, #00ff9d, #00d1ff) !important;
        color: black !important; font-weight: bold; border-radius: 18px; height: 4em; font-size: 28px !important;
        border: none; box-shadow: 0 0 30px #00ff9d88; transition: all 0.4s;
    }
    .stButton > button:hover {transform: scale(1.08); box-shadow: 0 0 50px #00ff9d !important;}

    .metric-card {
        background: linear-gradient(145deg, #1a1f2e, #16213e);
        padding: 1.8rem; border-radius: 18px; border: 2px solid #00ff9d33;
        text-align: center; box-shadow: 0 10px 40px rgba(0,255,150,0.2);
    }
    
    /* Mejoras para métricas */
    .stMetric {
        background: linear-gradient(145deg, #1a1f2e, #16213e) !important;
        padding: 1.5rem !important;
        border-radius: 18px !important;
        border: 3px solid #00ff9d !important;
        box-shadow: 0 0 30px rgba(0, 255, 157, 0.6) !important;
        transition: all 0.3s ease !important;
    }

    .stMetric:hover {
        box-shadow: 0 0 50px rgba(0, 255, 157, 0.9) !important;
        transform: translateY(-5px) !important;
    }

    /* Valores de las métricas bien grandes y blancos */
    .css-1x8b8vq, .css-1d391kg, .stMetric > div > div > div {
        color: white !important;
        font-size: 2rem !important;
        font-weight: bold !important;
    }

    /* Etiquetas de las métricas en verde neón */
    .stMetric label {
        color: #00ff9d !important;
        font-size: 1.2rem !important;
        font-weight: bold !important;
    }

    /* Botón de navegación flotante */
    .floating-nav {
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 999;
    }

    /* TÍTULO "MODELOS A FINE-TUNEAR" MÁS GRANDE Y BLANCO */
    div[data-testid="stMultiSelect"] label {
        font-size: 28px !important;
        color: white !important;
        font-weight: bold !important;
        margin-bottom: 10px !important;
    }
     
    /* SUBTÍTULO "Mejora tus modelos..." - VERSIÓN MÁS ESPECÍFICA */
    div[data-testid="stMarkdown"] p:first-of-type {
        font-size: 18px !important;
        color: white !important;
        font-weight: bold !important;
    }

    /* TEXTO "Intensidad del Fine-Tuning" MÁS GRANDE Y BLANCO */
    div[data-testid="stSlider"] label {
        font-size: 22px !important;
        color: white !important;
        font-weight: bold !important;
    }

    /* TEXTO "Datos hasta (recomendado: hoy)" MÁS GRANDE Y BLANCO */
    div[data-testid="stDateInput"] label {
        font-size: 22px !important;
        color: white !important;
        font-weight: bold !important;
    }

    /* TEXTO "Horizonte de predicción" MÁS GRANDE Y BLANCO */
    div[data-testid="stSelectbox"] label {
        font-size: 22px !important;
        color: white !important;
        font-weight: bold !important;
    }

</style>
""", unsafe_allow_html=True)

# TÍTULO ÉPICO MEJORADO
st.markdown("""
<div style='text-align:center; padding:3rem; background:linear-gradient(90deg,#000428,#004e92); border-radius:25px; border:3px solid #00ff9d; margin-bottom:2rem;'>
    <h1 style='font-size:4.5rem; margin:0;'>🤖 ETH GRIDBOT AI 2025</h1>
    <p style='color:#00d1ff; font-size:1.8rem; margin:10px;'>4 Modelos Avanzados • EGARCH+FHS • Backtesting 24x • Fine-Tuning Real</p>
</div>
""", unsafe_allow_html=True)

# ========================================
# PESTAÑAS
# ========================================

tab1, tab2, tab3 = st.tabs([
    "🎯 1. Descubrir Mejor Estrategia (24 combinaciones)",
    "⚡ 2. Fine-Tuning Avanzado", 
    "🚀 3. Predicción Oficial"
])

# ===================================================================
# TAB 1: Descubrir Mejor Estrategia (24 combinaciones)
# ===================================================================
with tab1:
    st.markdown("### 🎯 Backtesting Completo - 24 Combinaciones")
    st.info("Diaria/Semanal × Mensual/Trimestral/Semestral × 4 Modelos")

    # BOTÓN  
    if st.button("🚀 EJECUTAR BACKTESTING COMPLETO (24 COMBINACIONES)", type="primary", use_container_width=True, key="btn_backtesting"):
        with st.spinner("🔄 Ejecutando 24 backtests simultáneos..."):
            # ... (tu código existente aquí)


            results_df = run_full_backtest()
            if not results_df.empty:
                st.success("BACKTESTING COMPLETADO")
                st.dataframe(results_df.style.highlight_max(subset=['Correlación'], color='#90EE90').format({'Correlación': '{:.4f}'}), use_container_width=True)
                mejor = results_df.iloc[0]
                st.markdown(f"""
                <div style='text-align:center; padding:2rem; background:#1a1f2e; border:3px solid #00ff9d; border-radius:20px;'>
                    <h2>MEJOR CONFIGURACIÓN</h2>
                    <h1 style='color:#00ff9d'>{mejor['Modelo']} • {mejor['Frecuencia']} • {mejor['Horizonte']}</h1>
                    <h3>Correlación: {mejor['Correlación']:.4f}</h3>
                </div>
                """, unsafe_allow_html=True)
                # Guardar mejor config
                best = {
                    "model": mejor['Modelo'],
                    "freq": "1D" if mejor['Frecuencia'] == 'Diaria' else "7D",
                    "horizon": int(mejor['Horizonte_días']),
                    "correlation": float(mejor['Correlación']),
                    "date": datetime.now().strftime("%Y-%m-%d")
                }
                with open("./models/best_config.json", "w") as f:
                    json.dump(best, f, indent=2)
            else:
                st.error("Error en backtesting")

# ===================================================================
# TAB 2: FINE-TUNING AVANZADO (Datos diarios · Fine-tuning real)
# ===================================================================
with tab2:
    st.markdown("### ⚡ Fine-Tuning Avanzado de Modelos")
    st.markdown("**Mejora tus modelos existentes con datos nuevos • Solo disponible si ya están inicializados**")

    import os
    from datetime import datetime

    # Archivos requeridos para considerar un modelo "listo para fine-tuning"
    REQUIRED_FILES = {
        "Modelo 1 (XGB+QR)": ["./models/model1_xgb.pkl", "./models/model1_qr.pkl"],
        "Modelo 2 (LSTM+Transformer)": ["./models/model2_lstm.keras"],
        "Modelo 3 (Hybrid)": ["./models/model3_lgb.pkl", "./models/model3_tft.keras"],
        "Modelo 4 (ARIMA+GRU)": ["./models/model4_gru.keras", "./models/model4_arima.pkl"],
        "Modelo 5 (EGARCH Volatilidad)": ["./models/egarch_fitted.pkl"]
    }

    # Detectar modelos listos
    ready_models = []
    missing_models = []

    for name, files in REQUIRED_FILES.items():
        if all(os.path.exists(f) for f in files):
            ready_models.append(name)
        else:
            missing_models.append(name)

    # Estado visual
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Listos para Fine-Tuning", len(ready_models), delta=f"{len(ready_models)}/5")
    with col2:
        st.metric("Faltan pesos base", len(missing_models))
    with col3:
        #st.markdown(f"**{datetime.now().strftime('%d %B %Y')}**")
        st.write("") 
    if missing_models:
        st.error("Los siguientes modelos **NO** pueden hacer fine-tuning todavía:")
        for m in missing_models:
            st.write(f"• {m}")
        st.info("→ Usa **Inicializar Modelos** o genera una predicción oficial primero.")

    if not ready_models:
        st.stop()

    # === SELECTOR DE INTENSIDAD ===
    intensity = st.select_slider(
        "Intensidad del Fine-Tuning",
        options=["Rápido (5-10 min)", "Equilibrado (15-30 min)", "Profundo (40-90 min)", "Extremo (2h+)"],
        value="Equilibrado (15-30 min)",
        help="Más intenso = más épocas = mejor adaptación al mercado actual"
    )

    # Épocas por intensidad
    epochs_config = {
        "Rápido (5-10 min)":       {"m2": 50,  "m3": 50,  "m4": 80},
        "Equilibrado (15-30 min)": {"m2": 120, "m3": 100, "m4": 200},
        "Profundo (40-90 min)":    {"m2": 250, "m3": 180, "m4": 400},
        "Extremo (2h+)":           {"m2": 500, "m3": 350, "m4": 800}
    }
    epochs = epochs_config[intensity]

    # Selección de modelos
    selected_models = st.multiselect(
        "Modelos a fine-tunear",
        options=ready_models,
        default=[]
    )


    col_a, col_b = st.columns([3, 2])
    with col_a:
        btn_selected = st.button("🎯 INICIAR FINE-TUNING SELECCIONADOS", type="primary", use_container_width=True, key="btn_finetune_selected")
    with col_b:
        btn_all = st.button("🔥 TODOS LOS MODELOS", type="secondary", use_container_width=True, key="btn_finetune_all")


    if btn_all:
        selected_models = ready_models.copy()
        st.success(f"→ Fine-Tuning **{intensity}** en {len(selected_models)} modelos")

    cutoff_date = st.date_input("Datos hasta (recomendado: hoy)", datetime.today())

    # === EJECUCIÓN ===
    if (btn_selected or btn_all) and selected_models:
        st.markdown(f"### Fine-Tuning {intensity} en curso...")
        
        with st.spinner("Mejorando modelos con datos frescos..."):
            df_full = get_full_dataset()
            df_train = df_full.loc[:str(cutoff_date)]

            progress = st.progress(0)
            log_box = st.empty()
            success_count = 0

            for i, model_name in enumerate(selected_models):
                log_box.text(f"Procesando → {model_name}")

                try:
                    if model_name == "Modelo 1 (XGB+QR)":
                        for f in REQUIRED_FILES[model_name]:
                            if os.path.exists(f): os.remove(f)
                        from src.models.model1_xgb_qr import predict_model1
                        predict_model1(df_train, df_train)
                        st.success(f"{model_name} → Reentrenado completo")

                    elif model_name == "Modelo 2 (LSTM+Transformer)":
                        from src.models.model2_lstm_transformer import finetune_model2
                        finetune_model2(df_train, epochs=epochs["m2"])
                        st.success(f"{model_name} → {epochs['m2']} épocas")

                    elif model_name == "Modelo 3 (Hybrid)":
                        from src.models.model3_hybrid import finetune_hybrid
                        finetune_hybrid(df_train, epochs_tft=epochs["m3"])
                        st.success(f"{model_name} → {epochs['m3']} épocas")

                    elif model_name == "Modelo 4 (ARIMA+GRU)":
                        from src.models.model4_arima_gru import finetune_gru
                        finetune_gru(df_train, epochs=epochs["m4"])
                        st.success(f"{model_name} → GRU {epochs['m4']} épocas + ARIMA nuevo")

                    elif model_name == "Modelo 5 (EGARCH Volatilidad)":
                        returns = np.log(df_train["ETH-USD_Close"] / df_train["ETH-USD_Close"].shift(1)).dropna()
                        from src.models.volatility import fit_egarch_high_quality
                        fit_egarch_high_quality(returns)
                        st.success(f"{model_name} → Reentrenado")

                    success_count += 1

                except Exception as e:
                    st.error(f"Error en {model_name}: {e}")

                progress.progress((i + 1) / len(selected_models))

            log_box.empty()
            progress.empty()

        st.balloons()
        st.success(f"¡FINE-TUNING COMPLETADO!")
        st.info(f"{success_count}/{len(selected_models)} modelos mejorados con datos hasta **{cutoff_date.strftime('%d %B %Y')}**")
        st.caption("Tu GridBot ahora está perfectamente adaptado al mercado actual")


# ===================================================================
# TAB 3: PREDICCIÓN OFICIAL (usa el ganador del backtesting)
# ===================================================================
with tab3:
    st.markdown("### 🚀 Predicción Oficial")
    st.markdown("**Usa automáticamente la mejor configuración encontrada en Backtesting**")

    # Cargar mejor configuración
    default_config = {
        "model": "Modelo 2 (LSTM+Transformer)",
        "freq": "1D",
        "horizon": 90,
        "correlation": 0.0,
        "date": "Nunca"
    }
    
    if os.path.exists("./models/best_config.json"):
        try:
            with open("./models/best_config.json", "r") as f:
                best = json.load(f)
            st.success(f"Usando MEJOR CONFIGURACIÓN del backtesting:\n"
                       f"**{best.get('model', 'Desconocido')}** • "
                       f"{best.get('freq', '1D')} • {best.get('horizon', 90)} días "
                       f"(correlación: {best.get('correlation', 0):.4f} - {best.get('date', 'hoy')})")
        except:
            best = default_config
            st.warning("Archivo best_config.json corrupto → usando valores por defecto")
    else:
        best = default_config
        st.warning("No se ha ejecutado Backtesting aún → usando Modelo 2 por defecto")

    # Horizonte (puedes cambiarlo manualmente si quieres)
    horizon_options = [30, 90, 180]
    default_index = horizon_options.index(best["horizon"]) if best["horizon"] in horizon_options else 1
    horizon = st.selectbox("Horizonte de predicción", horizon_options, index=default_index)

    cutoff_date = st.date_input("Fecha de predicción (hoy)", datetime.today())
    prediction_date = cutoff_date + timedelta(days=horizon)



    if st.button("🎯 GENERAR PREDICCIÓN OFICIAL CON IA", type="primary", use_container_width=True, key="btn_predict"):
        with st.spinner("🧠 Generando predicción oficial con IA..."):


            df_full = get_full_dataset()
            df_train = df_full.loc[:str(cutoff_date)]

            # Seleccionar el modelo GANADOR del backtesting
            model_name = best.get("model", "Modelo 2 (LSTM+Transformer)")
            if "XGB" in model_name or "Modelo 1" in model_name:
                final_price = predict_model1(df_train, df_train)
            elif "LSTM" in model_name or "Modelo 2" in model_name:
                final_price = predict_model2(df_train, df_train)
            elif "Hybrid" in model_name or "Modelo 3" in model_name:
                final_price = predict_model3(df_train, df_train)
            else:
                final_price = predict_model4(df_train, df_train)

            # Volatilidad EGARCH
            returns = np.log(df_train["ETH-USD_Close"] / df_train["ETH-USD_Close"].shift(1)).dropna()
            egarch = load_joblib("./models/egarch_fitted.pkl") or fit_egarch_high_quality(returns)
            pcts = simulate_fhs_egarch_v2(egarch, final_price, horizon=horizon)
            low_70, high_70 = int(pcts[2]), int(pcts[4])
            low_90, high_90 = int(pcts[1]), int(pcts[5])

            # Guardar predicción
            pred_data = {
                "prediction_date": prediction_date.strftime("%Y-%m-%d"),
                "expected_price": int(final_price),
                "range_70pct": [low_70, high_70],
                "range_90pct": [low_90, high_90],
                "model_used": model_name,
                "backtest_correlation": best.get("correlation", 0),
                "generated_at": datetime.now().isoformat()
            }
            with open("predicciones.json", "w") as f:
                json.dump(pred_data, f, indent=2)

        st.balloons()
        st.success(f"Predicción generada para {prediction_date.strftime('%d %B %Y')}")
        


        # METRICAS FINALES:
        
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "💰 Precio Esperado", 
                f"${int(final_price):,}",
                delta="Predicción IA"
            )

        with col2:
            st.metric(
                "🎯 Rango 70% Confianza",
                f"${low_70:,} - ${high_70:,}",
                delta=f"±{((high_70-low_70)/final_price*100/2):.1f}%"
            )

        with col3:
            st.metric(
                "📊 Rango 90% Confianza", 
                f"${low_90:,} - ${high_90:,}",
                delta=f"±{((high_90-low_90)/final_price*100/2):.1f}%"
            )

        with col4:
            st.metric(
                "🤖 Grid Sugerido",
                f"${int(final_price*0.74):,} - ${high_70:,}",
                delta="Para GridBot"
            )

# ========================================
# BOTÓN PARA VOLVER AL LAUNCHER 
# ========================================
import streamlit as st
import os
import subprocess
import sys
import time
import webbrowser

# Ruta absoluta del launcher 
LAUNCHER_PATH = r"D:\Users\Usuario\Desktop\maestria\Proyecto de grado\ETH-USD-GridBot\app_launcher.py"

# Botón fijo 
st.markdown("""
<style>
    .fixed-back {
        position: fixed;
        bottom: 20px;
        left: 20px;
        z-index: 999999;
        background: linear-gradient(90deg, #ff2a6d, #d100d1);
        padding: 2px;
        border-radius: 15px;
    }
</style>
<div class="fixed-back">
""", unsafe_allow_html=True)

if st.button("VOLVER AL MENÚ PRINCIPAL", use_container_width=True, key="back_to_launcher"):
    st.balloons()
    st.success("Regresando al ETH GridBot AI 2025...")
    
    dir_launcher = os.path.dirname(LAUNCHER_PATH)
    file_launcher = os.path.basename(LAUNCHER_PATH)
    
    cmd = [sys.executable, "-m", "streamlit", "run", file_launcher, "--server.port", "8501"]
    
    if sys.platform.startswith("win"):
        subprocess.Popen(cmd, cwd=dir_launcher, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        subprocess.Popen(cmd, cwd=dir_launcher)
    
    time.sleep(2.3)
    webbrowser.open("http://localhost:8501")
    
    st.info("¡Launcher abierto! Puedes cerrar esta pestaña cuando quieras")

st.markdown("</div>", unsafe_allow_html=True)




# =============================================
# 🔁 BLOQUE: Rolling Expanding Forecast (UI BONITA)
# =============================================
from datetime import date
import tempfile
import shutil
import streamlit as st
import pandas as pd
import os
from src.rolling_forecast import rolling_expanding_forecast

st.markdown("""
<style>
.card {
    padding: 15px;
    border-radius: 10px;
    background-color: #111827;
    border: 1px solid #374151;
    margin-bottom: 12px;
}
.title {
    font-size: 24px;
    font-weight: bold;
}
.subtitle {
    font-size: 13px;
    opacity: 0.6;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🔁 **Simulación Expansiva (Rolling / Expanding Forecast)** ")
st.markdown("<div class='subtitle'>Ejecuta predicciones trimestrales sin fuga de información.</div>", unsafe_allow_html=True)
st.markdown("---")

with st.expander("⚙️ **Configuración de la Simulación**", expanded=False):

    # -----------------------------
    # TARJETAS INPUTS
    # -----------------------------
    colA, colB = st.columns(2)

    with colA:
        st.markdown("<div class='card'><b>📅 Fechas de Entrada</b></div>", unsafe_allow_html=True)
        fecha_descarga = st.date_input("Fecha inicio descarga", value=date(2022,1,1))
        fecha_inicio = st.date_input("Fecha inicio backtesting", value=date(2023,1,1))

    with colB:
        st.markdown("<div class='card'><b>⏳ Horizonte & Modelo</b></div>", unsafe_allow_html=True)
        fecha_fin = st.date_input("Fecha fin backtesting", value=date(2024,1,1))
        horizonte = st.selectbox("Horizonte", options=["mensual", "trimestral", "semestral"], index=1)

    st.markdown("### 🤖 **Modelo a usar**")
    modelo = st.selectbox(
        "",
        options=[
            "model1_xgb_qr",
            "model2_lstm_transformer",
            "model3_tabnet_tft",
            "model4_arima_gru"
          
        ],
        index=1
    )

    # -----------------------------
    # BOTONES
    # -----------------------------
    json_path_default = "rolling_results.json"

    colRun, colClear = st.columns([3,1])

    with colRun:
        run_btn = st.button("▶️ Ejecutar Simulación Expansiva", use_container_width=True)

    with colClear:
        clear_btn = st.button("🗑️ Limpiar JSON previo", use_container_width=True)

    # -----------------------------
    # BORRAR JSON
    # -----------------------------
    if clear_btn:
        if os.path.exists(json_path_default):
            os.remove(json_path_default)
            st.success("🗑️ JSON anterior eliminado.")
        else:
            st.info("No existía archivo JSON previo.")

    # -----------------------------
    # EJECUTAR ROLLING
    # -----------------------------
    if run_btn:

    
        # LIMPIAR JSON ANTES DE NUEVA EJECUCIÓN
        if os.path.exists(json_path_default):
            os.remove(json_path_default)
            st.info("🗑️ Archivo anterior eliminado. Comenzando nueva ejecución...")
        # ================================


        st.markdown("### 🚀 Ejecutando Rolling Forecast…")
        with st.spinner("Procesando… esto puede tardar si usas EGARCH..."):

            resultados = rolling_expanding_forecast(
                fecha_descarga_inicio=str(fecha_descarga),
                fecha_inicio_backtest=str(fecha_inicio),
                fecha_fin_backtest=str(fecha_fin),
                horizonte=horizonte,
                modelo_id=modelo,
                save_json_path=json_path_default,
                egarch_path="./models/egarch_fitted.pkl",
                min_rows_required=200,
                n_sims_egarch=5000
            )

        # -----------------------------
        
                # -----------------------------
        # MOSTRAR RESULTADOS
        # -----------------------------
        if resultados:

            st.success(f"🎉 Simulación completada — {len(resultados)} predicciones generadas.")

            # Convertir resultados a DataFrame
            df_out = pd.DataFrame(resultados)

            st.markdown("### 📊 **Últimas predicciones (vista estilo Cyber)**")

            # ---------------------------------------
            # TABLA ESTILO CYBER (idéntica a tu diseño)
            # ---------------------------------------
            import streamlit.components.v1 as components
            import html

            # Columnas esperadas
            expected_cols = [
                "id",
                "fecha_inicio_corte",
                "fecha_predicha",
                "pred_close",
                "pred_high",
                "pred_low",
                "tendencia"
            ]
            for c in expected_cols:
                if c not in df_out.columns:
                    df_out[c] = ""

            def fmt_money(x):
                try:
                    return f"${float(x):,.2f}"
                except:
                    return "-"

            def clean_txt(x):
                return html.escape(str(x))

            html_table = """
            <style>
            .cyber-table {font-family: 'Courier New', monospace; width: 100%; border-collapse: collapse; background: #0a0e17; border-radius: 12px; overflow: hidden; box-shadow: 0 0 40px rgba(0, 255, 157, 0.35); border: 1px solid rgba(0,255,157,0.08);}
            .cyber-table thead th {background: linear-gradient(90deg, #00d1ff, #00ff9d); color: #000; padding: 14px; text-align: center; font-weight: 700; font-size: 18px;}
            .cyber-table tbody td {padding: 12px; text-align: center; color: #e6f7ff; font-size: 15px; border-top: 1px solid rgba(255,255,255,0.03);}
            .cyber-table tbody tr:nth-child(even) {background: rgba(0, 255, 157, 0.03);}
            .cyber-table tbody tr:hover {background: linear-gradient(90deg, rgba(0,209,255,0.06), rgba(0,255,157,0.04)); transform: translateY(-1px); box-shadow: 0 8px 30px rgba(0,255,157,0.06);}
            .up {color: #00ff9d; font-weight: 700;}
            .down {color: #ff4d73; font-weight: 700;}
            .neutral {color: #9aa4b2; font-weight: 600;}
            .table-wrap {max-height:520px; overflow:auto; padding-right:8px; border-radius:10px;}
            .header-note {color:#cfefff; font-size:13px; margin-bottom:6px;}
            </style>

            #<div class='header-note'>Resultados generados — vista estilo Cyberpunk. Incluye tendencia y valores High/Low por EGARCH+FHS.</div>
            <div class='table-wrap'>
            <table class='cyber-table'>
            <thead>
            <tr>
                <th>#</th>
                <th>Fecha Corte</th>
                <th>Fecha Predicha</th>
                <th>Close</th>
                <th>High</th>
                <th>Low</th>
                <th>Tendencia</th>
            </tr>
            </thead>
            <tbody>
            """

            for _, row in df_out.iterrows():
                idv = clean_txt(row["id"])
                f_inicio = clean_txt(row["fecha_inicio_corte"])
                f_pred = clean_txt(row["fecha_predicha"])
                close = fmt_money(row["pred_close"])
                high = fmt_money(row["pred_high"])
                low = fmt_money(row["pred_low"])

                tendencia_raw = str(row["tendencia"]).lower()
                if "alc" in tendencia_raw:
                    tendencia_html = "<span class='up'>Alcista</span>"
                    close_td = f"<td style='font-weight:700; color:#00ff9d'>{close}</td>"
                elif "baj" in tendencia_raw:
                    tendencia_html = "<span class='down'>Bajista</span>"
                    close_td = f"<td style='font-weight:700; color:#ff4d73'>{close}</td>"
                else:
                    tendencia_html = "<span class='neutral'>Neutral</span>"
                    close_td = f"<td style='font-weight:700; color:#c0d6e4'>{close}</td>"

                html_table += f"""
                <tr>
                    <td>{idv}</td>
                    <td>{f_inicio}</td>
                    <td>{f_pred}</td>
                    {close_td}
                    <td>{clean_txt(high)}</td>
                    <td>{clean_txt(low)}</td>
                    <td>{tendencia_html}</td>
                </tr>
                """

            html_table += "</tbody></table></div>"

            components.html(html_table, height=620, scrolling=True)

            # ---------------------------------------
            # DESCARGA CSV
            # ---------------------------------------
            csv_all = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Descargar Predicciones (CSV)",
                data=csv_all,
                file_name="rolling_predicciones.csv",
                mime="text/csv",
                use_container_width=True
            )

            # ---------------------------------------
            # DESCARGA JSON
            # ---------------------------------------
            with open(json_path_default, "r") as f:
                data_bytes = f.read().encode("utf-8")

            st.download_button(
                "⬇️ Descargar JSON",
                data=data_bytes,
                file_name=json_path_default,
                mime="application/json",
                use_container_width=True
            )

        else:
            st.warning("⚠️ No se generaron predicciones. Revisa logs en la consola del servidor.")

        st.markdown("---")







