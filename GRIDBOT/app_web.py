# app_web.py
import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import streamlit.components.v1 as components
# =============================================
# IMPORTS
# =============================================

print("🔄 Cargando módulos desde carpeta actual...")

try:
    # Importar directamente desde la misma carpeta
    from data_loader import cargar_datos_yfinance
    from gridbot_core import (
        distribuir_capital, generar_grilla, asignar_montos_por_grid,
        inicializar_gridbot, simular_gridbot
    )
    from gridbot_trades import (
        generar_detalle_trades_nivel, calcular_evolucion_balance
    )
    from gridbot_metrics import generar_metricas_gridbot
    from gridbot_strategies import (
        personalizacion_gridbot_inyeccion, personalizacion_gridbot_inyeccion2, generar_rango_dinamico
    )
    from utils import log_gridbot_event
    
    print("✅ Todos los módulos cargados exitosamente!")
    
except ImportError as e:
    print(f"❌ Error en importación: {e}")
    st.error(f"Error crítico: {e}")
    st.stop()

# =============================================
# FUNCIONES DE DETECCIÓN DE FRECUENCIA 
# =============================================

def detectar_frecuencia_predicciones(predicciones):
    """
    Detecta automáticamente si las predicciones son mensuales, trimestrales o semestrales
    basándose en las fechas del JSON del rolling forecast.
    """
    if len(predicciones) < 2:
        return "trimestral"  # Default seguro si no hay suficientes datos
    
    try:
        # Analizar diferencia entre las primeras dos fechas consecutivas
        fecha1 = pd.to_datetime(predicciones[0]["fecha_inicio_corte"])
        fecha2 = pd.to_datetime(predicciones[1]["fecha_inicio_corte"])
        dias_diferencia = (fecha2 - fecha1).days
        
        # Determinar frecuencia basada en los días de diferencia
        if 25 <= dias_diferencia <= 35:  # Aprox 1 mes
            return "mensual"
        elif 85 <= dias_diferencia <= 95:  # Aprox 3 meses  
            return "trimestral"
        elif 175 <= dias_diferencia <= 185:  # Aprox 6 meses
            return "semestral"
        else:
            # Si no coincide con los patrones esperados, usar trimestral como fallback
            return "trimestral"
            
    except Exception as e:
        print(f"⚠️ Error detectando frecuencia: {e}. Usando trimestral por defecto.")
        return "trimestral"

def mapear_predicciones_a_mensual(predicciones, frecuencia_predicciones):
    """
    Convierte cualquier frecuencia de predicción a índices mensuales para el GridBot.
    Retorna un diccionario {índice_mensual: predicción}
    """
    mapeo_mensual = {}
    
    if frecuencia_predicciones == "mensual":
        # Caso más simple: 1 predicción por mes
        for i, pred in enumerate(predicciones):
            mapeo_mensual[i] = pred
    
    elif frecuencia_predicciones == "trimestral":
        # 1 predicción trimestral cubre 3 meses
        for i, pred in enumerate(predicciones):
            for mes_offset in range(3):  # 3 meses por trimestre
                indice_mensual = i * 3 + mes_offset
                mapeo_mensual[indice_mensual] = pred
    
    elif frecuencia_predicciones == "semestral":
        # 1 predicción semestral cubre 6 meses
        for i, pred in enumerate(predicciones):
            for mes_offset in range(6):  # 6 meses por semestre
                indice_mensual = i * 6 + mes_offset
                mapeo_mensual[indice_mensual] = pred
    
    print(f"✅ Mapeo creado: {frecuencia_predicciones} → mensual")
    print(f"   Predicciones originales: {len(predicciones)}")
    print(f"   Meses cubiertos: {len(mapeo_mensual)}")
    
    return mapeo_mensual

def cargar_predicciones_rolling(ruta_archivo="rolling_results.json"):
    """
    Carga las predicciones del rolling forecast y detecta automáticamente su frecuencia.
    """
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            predicciones = json.load(f)
        
        if not predicciones:
            st.error("❌ El archivo de predicciones está vacío")
            return None
        
        # Detectar frecuencia automáticamente
        frecuencia = detectar_frecuencia_predicciones(predicciones)
        
        # Crear mapeo para GridBot (siempre trabajamos con índices mensuales internamente)
        mapeo_predicciones = mapear_predicciones_a_mensual(predicciones, frecuencia)
        
        st.success(f"✅ Predicciones cargadas: {len(predicciones)} {frecuencia} → {len(mapeo_predicciones)} meses")
        st.info(f"📅 Frecuencia detectada: {frecuencia}")
        
        return mapeo_predicciones
        
    except FileNotFoundError:
        st.error(f"❌ No se encontró el archivo: {ruta_archivo}")
        return None
    except json.JSONDecodeError:
        st.error("❌ Error al decodificar el JSON")
        return None

# =============================================
# FUNCIONES AUXILIARES COMPLETAS
# =============================================

def cargar_predicciones(ruta_archivo="predicciones.json"):
    """Carga las predicciones desde el archivo JSON"""
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            predicciones = json.load(f)
            st.success("✅ Predicciones cargadas correctamente")
            return predicciones
    except FileNotFoundError:
        st.error(f"❌ No se encontró el archivo: {ruta_archivo}")
        st.info("💡 Ejecuta tu modelo de IA para generar el archivo 'predicciones.json'")
        return None
    except json.JSONDecodeError:
        st.error("❌ Error al leer el archivo JSON")
        return None

def crear_predicciones_ejemplo():
    """Crea predicciones de ejemplo si no existe el archivo"""
    return {
        "close_pred": 3520.75,
        "high_pred": 3850.25,
        "low_pred": 3150.50,
        "tendencia": "alcista",
        "confianza": 0.82,
        "timestamp": datetime.now().isoformat(),
        "modelo": "LSTM_v2",
        "variacion_close": 5.2
    }

def formatear_dinero(valor):
    """Formatea valores monetarios"""
    if pd.isna(valor) or valor is None:
        return "$0.00"
    return f"${float(valor):,.2f}"

def formatear_porcentaje(valor):
    """Formatea porcentajes"""
    if pd.isna(valor) or valor is None:
        return "0.00%"
    return f"{float(valor):.2f}%"

# =============================================
# CONFIGURACIÓN + TEMA CYBERPUNK 
# =============================================
st.set_page_config(
    page_title="GridBot IA 2025 - Trading Automatizado",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 
st.markdown("""
<style>
    .main > div {max-width: 95%; padding-top: 2rem;}
    .stApp {background: linear-gradient(135deg, #0a0e17, #1a0033);}
    
    h1, h2, h3 {font-family: 'Orbitron', sans-serif !important; color: #00c27a !important; text-shadow: 0 0 40px #00ff9d88;}
    
        
    /* Tabs como botones cyberpunk */
    .stTabs [data-baseweb="tab-list"] {gap: 20px; justify-content: center; flex-wrap: wrap;}
    .stTabs [data-baseweb="tab"] {
        height: 80px; width: 280px; border-radius: 18px; font-size: 20px; font-weight: bold;
        background: linear-gradient(145deg, #1a1f2e, #16213e); border: 3px solid #00ff9d44;
        color: #00d1ff !important; transition: all 0.4s;
    }
    .stTabs [data-baseweb="tab"]:hover {border: 3px solid #00ff9d; transform: translateY(-8px); box-shadow: 0 15px 35px #00ff9d44;}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(145deg, #00ff9d22, #00d1ff22); border: 3px solid #00ff9d; color: white !important;
    }

    /* Botones épicos */
    .stButton > button {
        background: linear-gradient(90deg, #00ff9d, #00d1ff) !important;
        color: black !important; font-weight: bold; border-radius: 18px; height: 4em; font-size: 22px !important;
        border: none; box-shadow: 0 0 30px #00ff9d88; transition: all 0.4s;
    }
    .stButton > button:hover {transform: scale(1.08); box-shadow: 0 0 50px #00ff9d !important;}

    .metric-card {
        background: linear-gradient(145deg, #1a1f2e, #16213e);
        padding: 1.8rem; border-radius: 18px; border: 2px solid #00ff9d33;
        text-align: center; box-shadow: 0 10px 40px rgba(0,255,150,0.2);
    }
    
    /* Mejoras adicionales para la versión completa */
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
        font-size: 1.8rem !important;
        font-weight: bold !important;
    }

    /* Etiquetas de las métricas en verde neón */
    .stMetric label {
        color: #00ff9d !important;
        font-size: 1.1rem !important;
        font-weight: bold !important;
    }


    /* TÍTULOS DEL SIDEBAR CON EL COLOR EXACTO DE PARÁMETROS GRIDBOT */
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5 {
        color: #00c27a !important;  /* Verde más oscuro como h1, h2, h3 */
        font-family: 'Orbitron', sans-serif !important;
        text-shadow: 0 0 40px #00ff9d88 !important;
        font-weight: bold !important;
        font-size: 19px !important; 
    }


</style>
""", unsafe_allow_html=True)

# TÍTULO 
st.markdown("""
<div style='text-align:center; padding:3rem; background:linear-gradient(90deg,#000428,#004e92); border-radius:25px; border:3px solid #00ff9d; margin-bottom:2rem;'>
    <h1 style='font-size:4rem; margin:0;'>🤖 GRIDBOT IA 2025</h1>
    <p style='color:#00d1ff; font-size:1.5rem; margin:10px;'>Inyecciones Inteligentes • Rango Dinámico con IA • Métricas Institucionales</p>
</div>
""", unsafe_allow_html=True)

# =============================================
# SESSION STATE
# =============================================
if 'df' not in st.session_state: st.session_state.df = None
if 'predicciones' not in st.session_state: st.session_state.predicciones = None
if 'predicciones_rolling' not in st.session_state: st.session_state.predicciones_rolling = None
if 'resultado_simulacion' not in st.session_state: st.session_state.resultado_simulacion = None
if 'page_number' not in st.session_state: st.session_state.page_number = 1


# =============================================
# FUNCIONES PRINCIPALES DE LA APP 
# =============================================

def mostrar_panel_principal():
    """Muestra el panel principal de la aplicación"""
    
    # Header informativo
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("📊 Panel de Control GridBot IA")
        st.markdown("""
        Esta aplicación combina:
        - **📈 Datos en tiempo real** de Yahoo Finance
        - **🤖 Predicciones de IA** para precios futuros  
        - **🤖 GridBot** para ejecución automática de trades
        - **📊 Análisis avanzado** de performance
        """)
    
    with col2:
        if st.session_state.df is not None:
            df = st.session_state.df
            precio_actual = df['close'].iloc[-1]
            if len(df) > 1:
                variacion = ((precio_actual - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
            else:
                variacion = 0
            
            st.metric(
                "Precio Actual",
                formatear_dinero(precio_actual),
                delta=formatear_porcentaje(variacion)
            )
        else:
            st.metric("Precio Actual", "Cargar datos", delta="---")
    
    with col3:
        if st.session_state.predicciones is not None:
            pred = st.session_state.predicciones
            st.metric(
                "Close Predicho",
                formatear_dinero(pred.get('close_pred', 0)),
                delta=pred.get('tendencia', 'N/A')
            )
        else:
            st.metric("Predicción IA", "Cargar datos", delta="---")
    
    # Mostrar datos cargados
    if st.session_state.df is not None:
        mostrar_datos_historicos(st.session_state.df)
    
    # Mostrar predicciones si existen
    if st.session_state.predicciones is not None:
        mostrar_predicciones_detalladas(st.session_state.predicciones)
    
    # Mostrar resultados de simulación si existen
    if st.session_state.resultado_simulacion is not None:
        mostrar_resultados_simulacion(st.session_state.resultado_simulacion)

def mostrar_datos_historicos(df):
    """Muestra los datos históricos cargados - VERSIÓN COMPLETA CORREGIDA"""
    
    st.subheader("📈 Datos Históricos Cargados")
    
    # Verificar y mostrar información completa del dataset
    st.write(f"**Información completa del dataset:**")
    
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.metric("Registros totales", f"{len(df):,}")
        st.metric("Precio Mínimo", formatear_dinero(df['close'].min()))
    
    with col_info2:
        st.metric("Fecha Inicial", df['fecha'].min().strftime('%Y-%m-%d'))
        st.metric("Precio Máximo", formatear_dinero(df['close'].max()))
    
    with col_info3:
        st.metric("Fecha Final", df['fecha'].max().strftime('%Y-%m-%d'))
        st.metric("Precio Actual", formatear_dinero(df['close'].iloc[-1]))
    
    # Mostrar estadísticas adicionales
    with st.expander("📊 Estadísticas Detalladas", expanded=False):
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            volatilidad = (df['close'].std() / df['close'].mean() * 100)
            st.metric("Volatilidad", f"{volatilidad:.1f}%")
        
        with col_stat2:
            cambio_total = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100)
            st.metric("Cambio Total", f"{cambio_total:.1f}%")
        
        with col_stat3:
            precio_medio = df['close'].mean()
            st.metric("Precio Medio", formatear_dinero(precio_medio))
        
        with col_stat4:
            rango_dias = (df['fecha'].max() - df['fecha'].min()).days
            st.metric("Período (días)", f"{rango_dias}")
    
    # Gráfico con formato de fechas en eje X
    fig = go.Figure()
    
    # Usar fechas reales en el eje X
    fig.add_trace(go.Scatter(
        x=df['fecha'],  # Fechas reales
        y=df['close'],
        mode='lines',
        name="Precio ETH/USDT",
        line=dict(color='#00C8FF', width=2),
        hovertemplate='<b>%{x|%Y-%m-%d %H:%M}</b><br>Precio: $%{y:.2f}<extra></extra>'
    ))
    
    # Configuración eje X  
    fig.update_layout(
        title="📈 Evolución del Precio - ETH/USDT",
        xaxis_title="Fecha",
        yaxis_title="Precio (USDT)",
        height=500,
        template="plotly_white",
        showlegend=False,
        xaxis=dict(
            type="date",
            tickformat="%b %Y",  # Formato: Ene 2020, Feb 2020, etc.
            dtick="M1",  # Un tick por mes
            tickangle=45,
            tickmode="auto",
            nticks=20,  # Máximo 20 ticks para evitar saturación
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            gridcolor='lightgray'
        )
    )
    
    # Ajustar automáticamente el espaciado de ticks según el rango temporal
    rango_dias = (df['fecha'].max() - df['fecha'].min()).days
    
    if rango_dias > 365 * 2:  # Más de 2 años
        fig.update_xaxes(dtick="M6", tickformat="%b %Y")  # Cada 6 meses
    elif rango_dias > 365:  # Más de 1 año
        fig.update_xaxes(dtick="M3", tickformat="%b %Y")  # Cada 3 meses
    elif rango_dias > 180:  # Más de 6 meses
        fig.update_xaxes(dtick="M1", tickformat="%b %Y")  # Cada mes
    elif rango_dias > 30:  # Más de 1 mes
        fig.update_xaxes(dtick="D7", tickformat="%d %b")  # Cada semana
    else:  # Menos de 1 mes
        fig.update_xaxes(dtick="D1", tickformat="%d %b")  # Cada día
    
    st.plotly_chart(fig, use_container_width=True)
    
    # SECCIÓN DE DATOS DETALLADOS 
    with st.expander("📋 Ver Datos Detallados (Tabla Completa)", expanded=False):
        st.write(f"**Mostrando todos los {len(df):,} registros del rango seleccionado**")
        
        # Crear DataFrame para mostrar
        df_display = df[['fecha', 'close']].copy()
        
        # Formatear fecha para mejor visualización
        df_display['fecha_str'] = df_display['fecha'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_display['close_usd'] = df_display['close'].apply(lambda x: f"${x:,.2f}")
        
        # Ordenar por fecha ascendente (desde el inicio hasta el final)
        df_display = df_display.sort_values('fecha', ascending=True)
        
        # Resetear índice para mostrar secuencia correcta
        df_display = df_display.reset_index(drop=True)
        df_display['#'] = df_display.index + 1
        
        # Mostrar tabla con controles de paginación
        st.write(f"**Registros del {df_display['fecha_str'].iloc[0]} al {df_display['fecha_str'].iloc[-1]}**")
        
        # Configurar paginación
        page_size = 100
        total_pages = max(1, len(df_display) // page_size + (1 if len(df_display) % page_size > 0 else 0))
        
        
        # Calcular rango para la página actual
        start_idx = (st.session_state.page_number - 1) * page_size
        end_idx = min(start_idx + page_size, len(df_display))
        
        # Mostrar información de paginación
        st.write(f"Mostrando registros {start_idx + 1} a {end_idx} de {len(df_display)} totales")
        
        # Crear subconjunto para mostrar
                # === TABLA DE PRECIOS  
        
                # === TABLA 100% FUNCIONAL (colores alternos + hover + flechas de tendencia)
        df_page = df_display.iloc[start_idx:end_idx].copy()
        df_page = df_page.reset_index(drop=True)
        df_page['#'] = df_page.index + start_idx + 1

        # Calcular tendencia respecto al precio anterior
        precios = df_page['close'].values
        tendencias = [""] * len(df_page)
        iconos = [""] * len(df_page)
        if len(precios) > 1:
            for i in range(1, len(precios)):
                if precios[i] > precios[i-1]:
                    tendencias[i] = "Sube"
                    iconos[i] = "Sube"
                elif precios[i] < precios[i-1]:
                    tendencias[i] = "Baja"
                    iconos[i] = "Baja"
                else:
                    tendencias[i] = "Igual"
                    iconos[i] = "Igual"

        df_page['Tendencia'] = iconos
        df_page['Precio Close (USD)'] = df_page['close_usd']
        df_page['Fecha y Hora'] = df_page['fecha_str']

        # Título 
        st.markdown(f"""
        <div style="text-align:center; margin:30px 0; padding:20px; background: linear-gradient(90deg, rgba(0,209,255,0.1), rgba(0,255,157,0.1)); border: 2px solid #00ff9d44; border-radius: 20px;">
            <h2 style="color:#00ff9d; text-shadow: 0 0 20px #00ff9d; margin:0;">
                HISTORIAL DE PRECIOS - ETH/USDT
            </h2>
            <p style="color:#00d1ff; font-size:1.4rem; margin:10px 0 0 0;">
                Registros {start_idx + 1} - {end_idx} de {len(df_display)} totales
            </p>
        </div>
        """, unsafe_allow_html=True)

        # TABLA CON components.html 
        import streamlit.components.v1 as components

        html = """
        <style>
            .cyber-table {font-family: 'Courier New', monospace; width: 100%; border-collapse: collapse;
                          background: #0a0e17; border-radius: 20px; overflow: hidden;
                          box-shadow: 0 0 40px rgba(0, 255, 157, 0.5);}
            .cyber-table th {background: linear-gradient(90deg, #00d1ff, #00ff9d); color: black;
                            padding: 16px; text-align: center; font-weight: bold; font-size: 18px;}
            .cyber-table td {padding: 14px; text-align: center; color: white; transition: all 0.3s;}
            .cyber-table tr:nth-child(even) {background: rgba(0, 255, 157, 0.08);}
            .cyber-table tr:hover {background: linear-gradient(90deg, rgba(0,209,255,0.2), rgba(0,255,157,0.2));
                                  transform: scale(1.01); box-shadow: 0 0 20px rgba(0,255,157,0.6);}
            .up   {color: #00ff9d; font-size: 28px; font-weight: bold;}
            .down {color: #ff0066; font-size: 28px; font-weight: bold;}
            .flat {color: #888; font-size: 24px;}
        </style>
        <table class="cyber-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Fecha y Hora</th>
                    <th>Precio Close (USD)</th>
                    <th>Tendencia</th>
                </tr>
            </thead>
            <tbody>
        """

        for _, row in df_page.iterrows():
            icon = ""
            if row['Tendencia'] == "Sube":
                icon = '<span class="up">Sube</span>'
            elif row['Tendencia'] == "Baja":
                icon = '<span class="down">Baja</span>'
            else:
                icon = '<span class="flat">Igual</span>'

            html += f"""
                <tr>
                    <td>{row['#']}</td>
                    <td>{row['Fecha y Hora']}</td>
                    <td style="font-weight: bold; color: #00ff9d;">{row['Precio Close (USD)']}</td>
                    <td>{icon}</td>
                </tr>
            """

        html += "</tbody></table>"

        # MOSTRAR LA TABLA 
        components.html(html, height=650, scrolling=True)
         
        
        # Controles de navegación 
         
        col_prev, col_info, col_next = st.columns([1, 2, 1])

        with col_prev:
            if st.session_state.page_number > 1:
                if st.button("◀️ Página Anterior", key="btn_prev_unique"):
                    st.session_state.page_number -= 1
                    st.rerun()
            else:
                st.write("")  # Mantiene el espacio

        with col_info:
            st.markdown(f"<h4 style='color:white; text-align:center;'>Página {st.session_state.page_number} de {total_pages}</h4>", 
                        unsafe_allow_html=True)

        with col_next:
            if st.session_state.page_number < total_pages:
                if st.button("Página Siguiente ▶️", key="btn_next_unique"):
                    st.session_state.page_number += 1
                    st.rerun()
            else:
                st.write("")  # Mantiene el espacio
        
        

        # Botones de descarga
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            # Descargar página actual
            csv_page = df_page.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Página Actual (CSV)",
                data=csv_page,
                file_name=f"datos_ethusd_pagina_{st.session_state.page_number}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_dl2:
            # Descargar dataset completo
            csv_full = df[['fecha', 'close']].to_csv(index=False)
            st.download_button(
                label="📥 Descargar Dataset Completo (CSV)",
                data=csv_full,
                file_name=f"datos_ethusd_completo_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Búsqueda por fecha
        st.markdown("---")
        st.subheader("🔍 Búsqueda por Fecha Específica")
        
        col_search1, col_search2 = st.columns(2)
        
        with col_search1:
            fecha_busqueda = st.date_input(
                "Seleccionar fecha para buscar",
                value=df['fecha'].min().date(),
                min_value=df['fecha'].min().date(),
                max_value=df['fecha'].max().date()
            )
        
        with col_search2:
            st.write("")  # Espacio
            if st.button("Buscar Fecha", use_container_width=True):
                # Buscar registros de esa fecha
                fecha_busqueda_dt = pd.to_datetime(fecha_busqueda)
                registros_fecha = df[df['fecha'].dt.date == fecha_busqueda]
                
                if len(registros_fecha) > 0:
                    st.success(f"✅ {len(registros_fecha)} registros encontrados para {fecha_busqueda}")
                    
                    # Mostrar resumen
                    df_resumen = registros_fecha[['fecha', 'close']].copy()
                    df_resumen['fecha_str'] = df_resumen['fecha'].dt.strftime('%H:%M:%S')
                    df_resumen['close_usd'] = df_resumen['close'].apply(lambda x: f"${x:,.2f}")
                    df_resumen = df_resumen.rename(columns={
                        'fecha_str': 'Hora',
                        'close_usd': 'Precio'
                    })
                    
                    st.dataframe(
                        df_resumen[['Hora', 'Precio']],
                        use_container_width=True,
                        height=200
                    )
                else:
                    st.warning(f"⚠️ No se encontraron registros para {fecha_busqueda}")

def mostrar_predicciones_detalladas(predicciones):
    """Muestra las predicciones de IA de forma detallada"""
    
    st.subheader("🤖 Predicciones de Inteligencia Artificial")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Close Predicho",
            formatear_dinero(predicciones.get('close_pred', 0)),
            delta=formatear_porcentaje(predicciones.get('variacion_close', 0))
        )
    
    with col2:
        st.metric("High Predicho", formatear_dinero(predicciones.get('high_pred', 0)))
    
    with col3:
        st.metric("Low Predicho", formatear_dinero(predicciones.get('low_pred', 0)))
    
    with col4:
        tendencia = predicciones.get('tendencia', 'neutral')
        color = "🟢" if tendencia == "alcista" else "🔴" if tendencia == "bajista" else "🟡"
        confianza = predicciones.get('confianza', 0) * 100
        st.metric("Tendencia", f"{color} {tendencia.title()}", delta=f"{confianza:.1f}% confianza")
    
    # Gráfico de predicciones
    fig = go.Figure()
    
    # Rango de predicción (High - Low)
    fig.add_trace(go.Bar(
        x=['Rango Predicho'],
        y=[predicciones.get('high_pred', 0) - predicciones.get('low_pred', 0)],
        base=predicciones.get('low_pred', 0),
        name='Rango High-Low Predicho',
        marker_color='lightblue',
        opacity=0.6,
        width=0.3
    ))
    
    # Close predicho
    fig.add_trace(go.Scatter(
        x=['Rango Predicho'],
        y=[predicciones.get('close_pred', 0)],
        mode='markers',
        name='Close Predicho',
        marker=dict(size=20, color='red', line=dict(width=2, color='darkred'))
    ))
    
    # Precio actual si está disponible
    if st.session_state.df is not None:
        precio_actual = st.session_state.df['close'].iloc[-1]
        fig.add_trace(go.Scatter(
            x=['Rango Predicho'],
            y=[precio_actual],
            mode='markers',
            name='Precio Actual',
            marker=dict(size=15, color='green', line=dict(width=2, color='darkgreen'))
        ))
    
    fig.update_layout(
        title="Análisis de Predicciones vs Precio Actual",
        yaxis_title="Precio (USDT)",
        showlegend=True,
        height=400,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# =============================================
# EJECUCIÓN DE LAS 3 ESTRATEGIAS
# =============================================

def ejecutar_simulacion_completa(
    df,
    predicciones,
    capital_inicial,
    gap_pct,
    modo_operacion,
    fee_pct,
    estrategia_capital,
    monto_inyeccion,
    frecuencia_inyeccion,
    usar_rango_ia,
    n_down,
    n_up,
    tolerancia_pct,
    precio_inferior_fijo=None,
    precio_superior_fijo=None
):
    """Ejecuta la simulación completa del GridBot - FUNCIÓN CORREGIDA"""
    
    st.header("🎯 Ejecutando Simulación del GridBot")
    
    # MOSTRAR INFORMACIÓN DE LA ESTRATEGIA SELECCIONADA
    if estrategia_capital == "Inversión Inicial":
        st.success("🟢 **ESTRATEGIA 1 - INVERSIÓN INICIAL**")
        st.info("• Grilla fija creada con primer precio histórico • Sin inyecciones adicionales • Backtesting básico puro")
    elif estrategia_capital == "Inyecciones Periódicas":
        st.success("💰 **ESTRATEGIA 2 - INYECCIONES PERIÓDICAS**")
        st.info(f"• Grilla fija • Inyecciones {frecuencia_inyeccion} de {formatear_dinero(monto_inyeccion)} • Distribución fija")
    else:  # Inyecciones con IA
        st.success("🤖 **ESTRATEGIA 3 - INYECCIONES CON IA**")
        st.info(f"• Grillas dinámicas cada 3 meses • Inyecciones {frecuencia_inyeccion} • Distribución adaptativa • IA ajusta rangos")

    with st.spinner("🔄 Configurando y ejecutando simulación..."):
        
        # 1. Determinar rango basado en IA o valores por defecto
        precio_actual = df['close'].iloc[-1]
        
        if usar_rango_ia and predicciones is not None:
            precio_predicho = predicciones.get('close_pred', precio_actual)
            
            precio_inferior, precio_superior = generar_rango_dinamico(
                df, None, 0, precio_actual, precio_predicho
            )
            
            st.success(f"🎯 **Rango Dinámico con IA:** {formatear_dinero(precio_inferior)} - {formatear_dinero(precio_superior)}")

        
            # Usar las predicciones mapeadas SI están disponibles
            if st.session_state.get('predicciones_rolling'):
                primera_pred = next(iter(st.session_state.predicciones_rolling.values()))
                tendencia_real = primera_pred.get('tendencia', 'neutral')
                precio_predicho_real = primera_pred.get('pred_close', precio_actual)
                st.info(f"📊 Basado en tendencia REAL: {tendencia_real} | Precio actual: {formatear_dinero(precio_actual)} | Predicción REAL: {formatear_dinero(precio_predicho_real)}")
            else:
                st.info(f"📊 Basado en tendencia: {predicciones.get('tendencia', 'neutral')} | Precio actual: {formatear_dinero(precio_actual)} | Predicción: {formatear_dinero(precio_predicho)}")
        
                     
        else:
            # PARA ESTRATEGIA 1 - USAR VALORES FIJOS DEL USUARIO
            if estrategia_capital == "Inversión Inicial" and precio_inferior_fijo is not None and precio_superior_fijo is not None:
                precio_inferior = precio_inferior_fijo
                precio_superior = precio_superior_fijo
                st.success(f"🎯 **Rango Fijo Estrategia 1:** {formatear_dinero(precio_inferior)} - {formatear_dinero(precio_superior)}")
                st.info("⚡ Backtesting puro con valores fijos del usuario")
            else:
                # Para otras estrategias - rangos automáticos
                rango_historico = df['close'].max() - df['close'].min()
                precio_inferior = max(precio_actual * 0.5, df['close'].min() * 0.8)
                precio_superior = min(precio_actual * 2.0, df['close'].max() * 1.2)
                st.success(f"📊 **Rango Automático:** {formatear_dinero(precio_inferior)} - {formatear_dinero(precio_superior)}")
        
        # 2. Inicializar GridBot
        try:
            estado_inicial = inicializar_gridbot(
                df=df,
                precio_superior=precio_superior,
                precio_inferior=precio_inferior,
                gap_pct=gap_pct,
                balance_inicial=capital_inicial,
                modo=modo_operacion
            )
            st.success("✅ GridBot inicializado correctamente")
        except Exception as e:
            st.error(f"❌ Error inicializando GridBot: {e}")
            return
        
        # 3. Mostrar configuración inicial
        mostrar_configuracion_inicial(estado_inicial)
        
        # 4. Ejecutar simulación según estrategia seleccionada 
        try:
            if estrategia_capital == "Inversión Inicial":
                st.info("🔧 Ejecutando ESTRATEGIA 1 - Simulación básica...")
                ordenes = simular_gridbot(
                    df, estado_inicial, n_down, n_up, tolerancia_pct, False
                )
                resultado = {
                    "ordenes": ordenes,
                    "detalle_trades": generar_detalle_trades_nivel(ordenes, estado_inicial, fee_pct),
                    "evolucion_balance": calcular_evolucion_balance(ordenes, estado_inicial, df, fee_pct),
                    "estado_final": estado_inicial,
                    "estrategia": "Inversión Inicial"
                }
                
          
            
            elif estrategia_capital == "Inyecciones Periódicas":
                st.info(f"💰 Ejecutando ESTRATEGIA 2 - Inyecciones {frecuencia_inyeccion} de {formatear_dinero(monto_inyeccion)}...")
    
                # PREPARAR PRECIOS PREDICHOS PARA ESTRATEGIA 2 
                precios_predichos = None
    
                if usar_rango_ia:
                    # PRIORIDAD 1: Usar predicciones mapeadas del rolling forecast
                    if st.session_state.get('predicciones_rolling'):
                        precios_predichos = st.session_state.predicciones_rolling
                        st.success(f"🎯 Estrategia 2 usando {len(precios_predichos)} predicciones mapeadas")
            
                        if precios_predichos:
                            primera_pred = next(iter(precios_predichos.values()))
                            if 'tendencia' in primera_pred:
                                st.info(f"📈 Primera predicción: {primera_pred['tendencia']} | Close: {formatear_dinero(primera_pred.get('pred_close', 0))}")
    
                resultado = personalizacion_gridbot_inyeccion(
                    df=df,
                    estado_inicial=estado_inicial,
                    monto_inyeccion=monto_inyeccion,
                    frecuencia=frecuencia_inyeccion,
                    fee_pct=fee_pct,
                    n_down=n_down,
                    n_up=n_up,
                    tolerancia_pct=tolerancia_pct,
                    mostrar_barra=False
                )
                resultado["estrategia"] = "Inyecciones Periódicas"






                
            else:  # Inyecciones con IA
                st.info(f"🧠 Ejecutando ESTRATEGIA 3 - Inyecciones {frecuencia_inyeccion} con IA...")
                
                # Preparar precios predichos para la simulación
                
                
                precios_predichos = None
                
                if usar_rango_ia:
                    # PRIORIDAD 1: Usar predicciones mapeadas del rolling forecast (si están disponibles)
                    if st.session_state.get('predicciones_rolling'):
                        precios_predichos = st.session_state.predicciones_rolling
                        st.success(f"🎯 Usando {len(precios_predichos)} predicciones mapeadas del rolling forecast")
                        
                        # Mostrar información sobre el mapeo
                        if precios_predichos:
                            primera_pred = next(iter(precios_predichos.values()))
                            if 'tendencia' in primera_pred:
                                st.info(f"📈 Primera predicción: {primera_pred['tendencia']} | Close: {formatear_dinero(primera_pred.get('pred_close', 0))}")
                    
                    # PRIORIDAD 2: Fallback a predicción simple individual (formato antiguo)
                    elif st.session_state.get('predicciones'):
                        prediccion_simple = st.session_state.predicciones
                        precio_predicho = prediccion_simple.get('close_pred', df['close'].iloc[0])
                        precios_predichos = {0: precio_predicho}
                        st.info("🔧 Usando predicción individual (formato simple)")
                    
                    # PRIORIDAD 3: Sin predicciones disponibles
                    else:
                        st.warning("⚠️ Modo IA activado pero no hay predicciones cargadas")
                        st.info("💡 Carga predicciones en el sidebar o desactiva 'Usar Rango Dinámico con IA'")
                        # Fallback seguro para evitar errores
                        precios_predichos = {0: df['close'].iloc[0]}

                # EJECUTAR ESTRATEGIA 3 CON LAS PREDICCIONES MEJORADAS
                resultado = personalizacion_gridbot_inyeccion2(
                    df=df,
                    estado_inicial=estado_inicial,
                    monto_inyeccion=monto_inyeccion,
                    frecuencia=frecuencia_inyeccion,
                    fee_pct=fee_pct,
                    n_down=n_down,
                    n_up=n_up,
                    tolerancia_pct=tolerancia_pct,
                    mostrar_barra=False,
                    funcion_rango=generar_rango_dinamico if usar_rango_ia else None,
                    precios_predichos=precios_predichos  # ← Pasa las predicciones mapeadas o el fallback
                )
                resultado["estrategia"] = "Inyecciones con IA"
            

            st.session_state.resultado_simulacion = resultado
            st.success(f"✅ Simulación {estrategia_capital} completada exitosamente!")
            st.success("Resultados guardados y visibles en la página principal")
            #st.info("Desplázate hacia abajo para ver todas las métricas, trades y gráficos")
            
        except Exception as e:
            st.error("Error durante la simulación:")
            st.code(str(e))
            import traceback
            st.code(traceback.format_exc(), language="python")
            st.warning("Hubo errores, pero los resultados parciales están disponibles")
            if 'resultado' in locals():
                st.session_state.resultado_simulacion = resultado             
                st.info("Resultados parciales guardados")
            return
        

def mostrar_configuracion_inicial(estado_inicial):
    """Muestra la configuración inicial del GridBot - VERSIÓN FINAL"""
    
    st.subheader("⚙️ Configuración del GridBot")
    
    # Crear dos columnas para mejor distribución
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(145deg, #1a1f2e, #16213e); 
                   padding: 1rem; border-radius: 12px; border: 2px solid #00ff9d33;
                   margin-bottom: 0.5rem;'>
            <h4 style='color: #00ff9d; text-align: center; margin-bottom: 0.8rem; font-size: 16px;'>💰 CAPITAL</h4>
        """, unsafe_allow_html=True)
        
        # Campos individuales con fondo gris y borde
        st.markdown("<div style='background: #2d3746; padding: 0.5rem; border-radius: 8px; border: 1px solid #00ff9d33; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)
        st.metric("💵 Precio Inicial", formatear_dinero(estado_inicial["precio_inicial"]), label_visibility="visible")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='background: #2d3746; padding: 0.5rem; border-radius: 8px; border: 1px solid #00ff9d33; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)
        st.metric("🏦 Capital Base", formatear_dinero(estado_inicial["capital_base"]), label_visibility="visible")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='background: #2d3746; padding: 0.5rem; border-radius: 8px; border: 1px solid #00ff9d33; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)
        st.metric("🪙 ETH Inicial", f"{estado_inicial['distribucion_capital']['balance_ETH']:.6f} ETH", label_visibility="visible")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='background: #2d3746; padding: 0.5rem; border-radius: 8px; border: 1px solid #00ff9d33; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)
        st.metric("💸 USDT Reservado", formatear_dinero(estado_inicial['distribucion_capital']['balance_USDT']), label_visibility="visible")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(145deg, #1a1f2e, #16213e); 
                   padding: 1rem; border-radius: 12px; border: 2px solid #00ff9d33;
                   margin-bottom: 0.5rem;'>
            <h4 style='color: #00ff9d; text-align: center; margin-bottom: 0.8rem; font-size: 16px;'>🎯 PARÁMETROS</h4>
        """, unsafe_allow_html=True)
        
        # Campos individuales con fondo gris y borde
        st.markdown("<div style='background: #2d3746; padding: 0.5rem; border-radius: 8px; border: 1px solid #00ff9d33; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)
        st.metric("🤖 Modo", estado_inicial['montos_por_grid']['modo'], label_visibility="visible")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='background: #2d3746; padding: 0.5rem; border-radius: 8px; border: 1px solid #00ff9d33; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)
        st.metric("🔢 Monto/Grid", f"{estado_inicial['montos_por_grid']['level_mount_grid']:.6f}", label_visibility="visible")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='background: #2d3746; padding: 0.5rem; border-radius: 8px; border: 1px solid #00ff9d33; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)
        st.metric("📏 Gap", f"{estado_inicial['niveles']['gap_pct']}%", label_visibility="visible")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='background: #2d3746; padding: 0.5rem; border-radius: 8px; border: 1px solid #00ff9d33; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)
        st.metric("🔢 Total Niveles", f"{len(estado_inicial['niveles']['niveles_totales'])}", label_visibility="visible")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='background: #2d3746; padding: 0.5rem; border-radius: 8px; border: 1px solid #00ff9d33; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)
        st.metric("⬇️ Compra", f"{len(estado_inicial['niveles']['niveles_down'])}", label_visibility="visible")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='background: #2d3746; padding: 0.5rem; border-radius: 8px; border: 1px solid #00ff9d33; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)
        st.metric("⬆️ Venta", f"{len(estado_inicial['niveles']['niveles_up'])}", label_visibility="visible")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Distribución en una fila completa con gris y borde
    st.markdown("""
    <div style='background: #2d3746; 
               padding: 1rem; border-radius: 12px; border: 2px solid #00ff9d33;
               margin-top: 0.5rem; text-align: center; width: 100%;'>
        <div style='display: flex; justify-content: center; align-items: center; gap: 2rem;'>
            <div style='color: #00ff9d; font-weight: bold; font-size: 16px;'>📊 DISTRIBUCIÓN</div>
            <div style='color: #e0e0e0; font-size: 16px; font-weight: bold;'>
                {}% ETH  -  {}% USDT
            </div>
        </div>
    </div>
    """.format(
        estado_inicial['distribucion_capital']['porcent_ETH'],
        estado_inicial['distribucion_capital']['porcent_USDT']
    ), unsafe_allow_html=True)


def mostrar_resultados_simulacion(resultado):
    """Muestra los resultados de la simulación - CORREGIDO SIN DUPLICACIONES"""
    
    # Mostrar información de la estrategia ejecutada
    estrategia = resultado.get("estrategia", "Desconocida")
    
    if estrategia == "Inversión Inicial":
        st.header("🟢 RESULTADOS - ESTRATEGIA 1: INVERSIÓN INICIAL")
    elif estrategia == "Inyecciones Periódicas":
        st.header("💰 RESULTADOS - ESTRATEGIA 2: INYECCIONES PERIÓDICAS")
    else:
        st.header("🤖 RESULTADOS - ESTRATEGIA 3: INYECCIONES CON IA")
    
    # Pestañas para organizar la información 
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Métricas Principales", 
        "💼 Detalle de Trades", 
        "📊 Evolución Balance",
        "🎯 Análisis Avanzado"
    ])
    
    with tab1:
        mostrar_metricas_principales(resultado)
    
    with tab2:
        mostrar_detalle_trades(resultado)
    
    with tab3:
        mostrar_evolucion_balance(resultado)
    
    with tab4:
        mostrar_analisis_avanzado(resultado)

def mostrar_metricas_principales(resultado):
    """Muestra las métricas principales de performance - SIN DUPLICACIONES"""
    
    # Calcular métricas
    detalle_trades = resultado["detalle_trades"]
    evolucion_balance = resultado["evolucion_balance"]
    
    try:
        metricas = generar_metricas_gridbot(detalle_trades, evolucion_balance, risk_free_rate=5.0)
        
        # Mostrar métricas en columnas  
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Profit Total",
                formatear_dinero(metricas.get("Profit Total (USDT)", 0)),
                delta=formatear_porcentaje(metricas.get("Retorno Total (%)", 0))
            )
            
            st.metric(
                "Win Rate",
                formatear_porcentaje(metricas.get("Win Rate (%)", 0))
            )
        
        with col2:
            st.metric(
                "Sharpe Ratio",
                f"{metricas.get('Sharpe Ratio', 0):.3f}"
            )
            
            st.metric(
                "Max Drawdown",
                formatear_porcentaje(metricas.get("Max Drawdown (%)", 0))
            )
        
        with col3:
            st.metric(
                "Trades Completados",
                f"{metricas.get('N° de Trades Completados', 0)}"
            )
            
            st.metric(
                "Profit Factor",
                f"{metricas.get('Profit Factor', 0):.2f}"
            )
        
        with col4:
            st.metric(
                "Volatilidad",
                formatear_porcentaje(metricas.get("Volatilidad (%)", 0))
            )
            
            st.metric(
                "Racha Máxima Ganadora",
                f"{metricas.get('Máx. Racha Ganadora', 0)}"
            )
        
        # Gráfico de evolución del balance  
        if not evolucion_balance.empty and 'valor_total_USDT' in evolucion_balance.columns:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=evolucion_balance.index,
                y=evolucion_balance['valor_total_USDT'],
                mode='lines',
                name='Valor Total',
                line=dict(color='green', width=3)
            ))
            
            # Línea de capital inicial
            capital_inicial = evolucion_balance['valor_total_USDT'].iloc[0]
            fig.add_hline(
                y=capital_inicial, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Capital Inicial: {formatear_dinero(capital_inicial)}"
            )
            
            fig.update_layout(
                title="Evolución del Valor del Portafolio",
                xaxis_title="Ticks de Simulación",
                yaxis_title="Valor Total (USDT)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ Error calculando métricas: {e}")

def mostrar_detalle_trades(resultado):
    """Muestra el detalle de todos los trades ejecutados"""
    
    st.subheader("📋 Detalle de Trades Ejecutados")
    
    detalle_trades = resultado["detalle_trades"]
    
    if not detalle_trades.empty:
        # Filtrar solo trades completados para métricas rápidas
        trades_completados = detalle_trades[detalle_trades['Estado'] == 'Completado']
        
        if not trades_completados.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                ganadores = len(trades_completados[trades_completados['Profit Neto (USDT)'] > 0])
                st.metric("Trades Ganadores", f"{ganadores}")
            
            with col2:
                perdedores = len(trades_completados[trades_completados['Profit Neto (USDT)'] < 0])
                st.metric("Trades Perdedores", f"{perdedores}")
            
            with col3:
                profit_promedio = trades_completados['Profit Neto (USDT)'].mean()
                st.metric("Profit Promedio", formatear_dinero(profit_promedio))
            
            with col4:
                eficiencia = ganadores / len(trades_completados) * 100
                st.metric("Eficiencia", f"{eficiencia:.1f}%")
        
        # Mostrar tabla completa
        st.dataframe(
            detalle_trades,
            use_container_width=True,
            height=400
        )
        
        # Botón para descargar
        csv = detalle_trades.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Detalle de Trades (CSV)",
            data=csv,
            file_name="detalle_trades_gridbot.csv",
            mime="text/csv"
        )
    else:
        st.info("📭 No hay trades ejecutados para mostrar")

def mostrar_evolucion_balance(resultado):
    """Muestra la evolución del balance en el tiempo"""
    
    st.subheader("💰 Evolución del Balance")
    
    evolucion_balance = resultado["evolucion_balance"]
    
    if not evolucion_balance.empty:
        # Gráfico de balances
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Evolución de Balances', 'Valor Total del Portafolio'),
            vertical_spacing=0.1
        )
        
        # Balances individuales
        if 'balance_ETH' in evolucion_balance.columns:
            fig.add_trace(
                go.Scatter(x=evolucion_balance.index, y=evolucion_balance['balance_ETH'], 
                          name='Balance ETH', line=dict(color='blue')),
                row=1, col=1
            )
        
        if 'balance_USDT' in evolucion_balance.columns:
            fig.add_trace(
                go.Scatter(x=evolucion_balance.index, y=evolucion_balance['balance_USDT'], 
                          name='Balance USDT', line=dict(color='orange')),
                row=1, col=1
            )
        
        # Valor total
        if 'valor_total_USDT' in evolucion_balance.columns:
            fig.add_trace(
                go.Scatter(x=evolucion_balance.index, y=evolucion_balance['valor_total_USDT'], 
                          name='Valor Total', line=dict(color='green', width=3)),
                row=2, col=1
            )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar tabla
        with st.expander("📋 Ver Evolución Completa en Tabla"):
            st.dataframe(
                evolucion_balance,
                use_container_width=True,
                height=300
            )
    else:
        st.info("📭 No hay datos de evolución de balance para mostrar")

def mostrar_analisis_avanzado(resultado):
    """Muestra análisis avanzados de la simulación"""
    
    st.subheader("🎯 Análisis Avanzado de Performance")
    
    detalle_trades = resultado["detalle_trades"]
    evolucion_balance = resultado["evolucion_balance"]
    
    # Análisis de distribución de profits
    trades_completados = detalle_trades[detalle_trades['Estado'] == 'Completado']
    
    if not trades_completados.empty and 'Profit Neto (USDT)' in trades_completados.columns:
        profits = trades_completados['Profit Neto (USDT)']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma de profits
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=profits,
                nbinsx=20,
                name='Distribución de Profits',
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title="Distribución de Profits por Trade",
                xaxis_title="Profit (USDT)",
                yaxis_title="Frecuencia",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot de profits
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=profits,
                name='Profits',
                boxpoints='outliers',
                marker_color='lightgreen'
            ))
            
            fig.update_layout(
                title="Distribución y Outliers de Profits",
                yaxis_title="Profit (USDT)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Análisis temporal
    if not evolucion_balance.empty and 'valor_total_USDT' in evolucion_balance.columns:
        st.subheader("📈 Análisis Temporal")
        
        # Calcular drawdown
        equity = evolucion_balance['valor_total_USDT']
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=evolucion_balance.index,
            y=drawdown * 100,
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='red'),
            fillcolor='rgba(255,0,0,0.3)'
        ))
        
        fig.update_layout(
            title="Evolución del Drawdown",
            xaxis_title="Ticks",
            yaxis_title="Drawdown (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# =============================================
# SIDEBAR COMPLETA - CON CAMPOS ESPECÍFICOS PARA ESTRATEGIA 1
# =============================================
with st.sidebar:
    st.markdown("<h2 style='color:#00ff9d; text-align:center;'>⚙️ CONFIGURACIÓN</h2>", unsafe_allow_html=True)
    
    # Configuración de datos
    st.markdown("<h3 style='color:#00d1ff;'>📊 Fuente de Datos</h3>", unsafe_allow_html=True)
    
    crypto_symbol = st.selectbox(
        "Criptomoneda", 
        ["ETH-USD", "BTC-USD", "ADA-USD", "DOT-USD", "LINK-USD"],
        index=0
    )
    
    col1, col2 = st.columns(2)
    with col1:
        fecha_inicio = st.date_input("Desde", datetime(2023, 1, 1))
    with col2:
        fecha_fin = st.date_input("Hasta", datetime.now())
    
    intervalo = st.selectbox(
        "Intervalo", 
        ["1d", "1h", "30m", "15m", "5m", "1m"], 
        index=0
    )
    
    if st.button("🔄 CARGAR DATOS EN TIEMPO REAL", type="primary", use_container_width=True):
        with st.spinner("Descargando y procesando datos..."):
            df = cargar_datos_yfinance(
                simbolo=crypto_symbol,
                fecha_inicio=fecha_inicio,
                fecha_fin=fecha_fin,
                intervalo=intervalo,
                force_refresh=True
            )
            st.session_state.df = df
            st.success("¡Datos cargados exitosamente!")
            st.balloons()

    st.markdown("---")
    

    
    # PREDICCIONES IA  
    st.markdown("<h3 style='color:#00d1ff;'>🤖 PREDICCIONES IA</h3>", unsafe_allow_html=True)

    usar_ejemplo = st.checkbox("Usar predicción de ejemplo", value=False)  # Cambiado a False

    if not usar_ejemplo:
        col_pred1, col_pred2 = st.columns([2, 1])
    
        with col_pred1:
            if st.button("📥 Cargar rolling_results.json", use_container_width=True, key="btn_cargar_rolling"):
                predicciones = cargar_predicciones_rolling("rolling_results.json")
                if predicciones:
                    st.session_state.predicciones_rolling = predicciones
                    # Tomar la primera predicción para mostrar en métricas del panel principal
                    primera_pred = next(iter(predicciones.values())) if predicciones else None
                    st.session_state.predicciones = primera_pred
                    st.success("✅ Predicciones del rolling forecast cargadas y mapeadas!")
    
        with col_pred2:
            if st.session_state.get('predicciones_rolling'):
                if st.button("🔄 Actualizar", use_container_width=True, key="btn_actualizar_rolling"):
                    st.rerun()
        
            # Botón alternativo para el formato antiguo (backward compatibility)
            if st.button("📁 Cargar predicciones.json", use_container_width=True, key="btn_cargar_antiguo"):
                predicciones_antiguas = cargar_predicciones("predicciones.json")
                if predicciones_antiguas:
                    st.session_state.predicciones = predicciones_antiguas
                    st.session_state.predicciones_rolling = None
                    st.success("✅ Predicciones en formato antiguo cargadas")
    else:
        st.session_state.predicciones = crear_predicciones_ejemplo()
        st.session_state.predicciones_rolling = None
        st.success("🗳️ Predicción de ejemplo activada")

    


    # Mostrar información de las predicciones cargadas
    if st.session_state.get('predicciones_rolling'):
        total_meses = len(st.session_state.predicciones_rolling)
        primera_pred = next(iter(st.session_state.predicciones_rolling.values()))
        precio_predicho = primera_pred.get('pred_close', 'N/A')
        tendencia = primera_pred.get('tendencia', 'N/A')
    
        st.success(f"📊 **Predicciones cargadas:** {total_meses} meses mapeados")
        st.info(f"🎯 **Primera predicción:** {formatear_dinero(precio_predicho)} | Tendencia: {tendencia}")
    
    elif st.session_state.get('predicciones'):
        st.info("📊 **Predicción simple cargada** (formato individual)")



    st.markdown("---")
    
    # Configuración del GridBot
    st.markdown("<h3 style='color:#00d1ff;'>🔧 PARÁMETROS GRIDBOT</h3>", unsafe_allow_html=True)
    
    col_cap, col_gap = st.columns(2)
    with col_cap:
        capital_inicial = st.number_input(
            "Capital Inicial (USDT)",
            min_value=100,
            max_value=100000,
            value=1000,
            step=100
        )
    with col_gap:
        gap_pct = st.slider(
            "Gap entre Niveles (%)",
            min_value=0.1,
            max_value=20.0,
            value=2.0,
            step=0.1
        )
    
    col_modo, col_fee = st.columns(2)
    with col_modo:
        modo_operacion = st.selectbox("Modo", ["ETH", "USDT"], index=0)
    with col_fee:
        fee_pct = st.slider("Fee (%)", 0.0, 1.0, 0.05, 0.01)
    
    # Configuración avanzada
    st.markdown("<h4 style='color:#00ff9d;'>🎯 Configuración Avanzada</h4>", unsafe_allow_html=True)
    
    usar_rango_ia = st.checkbox("Usar Rango Dinámico con IA", value=True)
    
    col_down, col_up = st.columns(2)
    with col_down:
        n_down = st.slider("Niveles Compra ↓", 1, 10, 3)
    with col_up:
        n_up = st.slider("Niveles Venta ↑", 1, 10, 3)
    
    tolerancia_pct = st.slider("Tolerancia Re-Compra (%)", 0.01, 5.0, 0.1, 0.01)
    
    # Estrategia de capital  
    st.markdown("<h4 style='color:#00ff9d;'>💰 Estrategia de Capital</h4>", unsafe_allow_html=True)
    
    estrategia_capital = st.radio(
        "Estrategia",
        ["Inversión Inicial", "Inyecciones Periódicas", "Inyecciones con IA"],
        index=0,
        label_visibility="collapsed"
    )
    
    # CAMPOS ESPECÍFICOS PARA ESTRATEGIA 1 - INVERSIÓN INICIAL
    if estrategia_capital == "Inversión Inicial":
        st.markdown("<h5 style='color:#00ff9d;'>🎯 Parámetros Específicos Estrategia 1</h5>", unsafe_allow_html=True)
        
        col_inf, col_sup = st.columns(2)
        with col_inf:
            precio_inferior_fijo = st.number_input(
                "Precio Inferior (USDT)",
                min_value=1,
                max_value=10000,
                value=7,   
                step=1,
                help="Precio mínimo del rango del grid"
            )
        with col_sup:
            precio_superior_fijo = st.number_input(
                "Precio Superior (USDT)", 
                min_value=100,
                max_value=50000,
                value=10000,   
                step=100,
                help="Precio máximo del rango del grid"
            )
        
        monto_inyeccion = 0
        frecuencia_inyeccion = "mensual"

    # CAMPOS PARA ESTRATEGIAS 2 Y 3
    elif estrategia_capital != "Inversión Inicial":
        monto_inyeccion = st.number_input(
            "Monto por Inyección (USDT)",
            min_value=10,
            max_value=5000,
            value=100,
            step=50
        )
        
        frecuencia_inyeccion = st.selectbox(
            "Frecuencia",
            ["diaria", "semanal", "mensual", "trimestral"],
            index=2
        )
    else:
        monto_inyeccion = 0
        frecuencia_inyeccion = "mensual"
        precio_inferior_fijo = 7
        precio_superior_fijo = 10000
    
    # Botón de ejecución principal
    st.markdown("---")
    
    
    ejecutar_simulacion = st.button("🚀 EJECUTAR SIMULACIÓN COMPLETA", 
                               type="primary", 
                               use_container_width=True,
                               key="ejecutar_simulacion_unico")



    if ejecutar_simulacion:
        # VERIFICAR que las predicciones siguen cargadas
        if st.session_state.get('predicciones_rolling') is None:
            st.error("❌ Las predicciones se perdieron. Recarga el JSON antes de ejecutar.")
            st.info("💡 Haz clic en '📥 Cargar rolling_results.json' nuevamente")
        else:
            st.success(f"✅ Predicciones confirmadas: {len(st.session_state.predicciones_rolling)} meses mapeados")
    
        if st.session_state.df is None:
            st.error("❌ Primero carga los datos históricos")
        elif st.session_state.predicciones is None:
            st.error("❌ Primero carga las predicciones de IA")
        else:

            # Pasar los parámetros específicos para Estrategia 1
             
            if estrategia_capital == "Inversión Inicial":
                # VERIFICAR que los valores se están pasando correctamente
                st.info(f"🎯 **Valores usados para Estrategia 1:**")
                st.info(f"Precio Inferior: {formatear_dinero(precio_inferior_fijo)}")
                st.info(f"Precio Superior: {formatear_dinero(precio_superior_fijo)}")
    
                ejecutar_simulacion_completa(
                    st.session_state.df,
                    st.session_state.predicciones,
                    capital_inicial,
                    gap_pct,
                    modo_operacion,
                    fee_pct,
                    estrategia_capital,
                    monto_inyeccion,
                    frecuencia_inyeccion,
                    False,  # Forzar NO usar rango IA para Estrategia 1
                    n_down,
                    n_up,
                    tolerancia_pct,
                    precio_inferior_fijo,
                    precio_superior_fijo
                )
            else:
                ejecutar_simulacion_completa(
                    st.session_state.df,
                    st.session_state.predicciones,
                    capital_inicial,
                    gap_pct,
                    modo_operacion,
                    fee_pct,
                    estrategia_capital,
                    monto_inyeccion,
                    frecuencia_inyeccion,
                    usar_rango_ia,
                    n_down,
                    n_up,
                    tolerancia_pct
                )

# =============================================
# MAIN APP - INTERFAZ PRINCIPAL
# =============================================

def main():
    """Función principal de la aplicación"""
    
    # Mostrar panel principal
    mostrar_panel_principal()


# =============================================
# BOTÓN FIJO PARA VOLVER AL LAUNCHER
# =============================================
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


# Ejecutar la aplicación
if __name__ == "__main__":
    main()