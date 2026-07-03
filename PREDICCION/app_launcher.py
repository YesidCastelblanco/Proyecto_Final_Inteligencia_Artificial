# launcher.py → LA VERSIÓN FINAL QUE NAVEGA ENTRE LAS 3 APPS
import streamlit as st
import os
import subprocess
import webbrowser
import time
import sys

st.set_page_config(page_title="ETH GridBot AI 2025 - Launcher", page_icon="🟢", layout="wide")

# Estilo

st.markdown("""
<style>
    .main > div {max-width: 95%; padding-top: 2rem;}
    .stApp {background: linear-gradient(135deg, #0a0e17, #1a0033);}
    h1, h2, h3 {font-family: 'Orbitron', sans-serif !important; color: #00c27a !important; text-shadow: 0 0 40px #00ff9d88;}
    .app-card {background: linear-gradient(145deg, #1a1f2e, #16213e); padding: 3rem; border-radius: 20px; 
               border: 3px solid #00ff9d44; margin: 2rem 0; text-align: center; transition: all 0.4s;}
    .app-card:hover {border: 3px solid #00ff9d; transform: translateY(-10px); box-shadow: 0 20px 40px #00ff9d44;}
    .stButton > button {background: linear-gradient(90deg, #00ff9d, #00d1ff) !important; color: black !important; 
                       font-weight: bold; border-radius: 18px; height: 4em; font-size: 24px !important;
                       border: none; box-shadow: 0 0 30px #00ff9d88; transition: all 0.4s; width: 100%;}
</style>
""", unsafe_allow_html=True)

# Título épico
st.markdown("""
<div style='text-align:center; padding:3rem; background:linear-gradient(90deg,#000428,#004e92); border-radius:25px; border:3px solid #00ff9d; margin-bottom:2rem;'>
    <h1 style='font-size:4.5rem; margin:0;'>🚀 ETH GRIDBOT AI 2025</h1>
    <p style='color:#00d1ff; font-size:1.8rem; margin:10px;'>Sistema Completo de Trading con IA</p>
</div>
""", unsafe_allow_html=True)

# RUTAS DE LOS PROYECTOS + RUTA DEL venv DE CADA UNO
APPS = {
    "predicciones": {
        "name": "PREDICCIONES IA",
        "script": r"D:\Users\Usuario\Desktop\maestria\Proyecto de grado\ETH-USD-GridBot\app.py",
        "venv":   r"D:\Users\Usuario\Desktop\maestria\Proyecto de grado\ETH-USD-GridBot\venv",  
        "description": "4 Modelos Avanzados • Backtesting • Fine-Tuning", 
        "features": [...],
        "port": 8502
    },
    "gridbot": {
        "name": "GRIDBOT TRADING",
        "script": r"D:\Users\Usuario\Desktop\maestria\Proyecto de grado\PROYECTO FINAL\GRIDBOT\app_web.py",
        "venv":   r"D:\Users\Usuario\Desktop\maestria\Proyecto de grado\PROYECTO FINAL\GRIDBOT\venv",  
        "description": "Trading Automatizado • 3 Estrategias • Simulación",
        "features": [...],
        "port": 8503
    }
}



def lanzar_con_venv(script_path: str, venv_path: str, puerto: int):
    if not os.path.exists(script_path):
        st.error(f"Script no encontrado:\n{script_path}")
        return
    if not os.path.exists(venv_path):
        st.error(f"Entorno virtual no encontrado:\n{venv_path}")
        return

    script_dir = os.path.dirname(script_path)
    script_file = os.path.basename(script_path)

    # RUTA DEL PYTHON DENTRO DEL VENV
    if sys.platform.startswith("win"):
        python_exe = os.path.join(venv_path, "Scripts", "python.exe")
        activate = os.path.join(venv_path, "Scripts", "activate.bat")
    else:
        python_exe = os.path.join(venv_path, "bin", "python")
        activate = os.path.join(venv_path, "bin", "activate")

    if not os.path.exists(python_exe):
        st.error(f"No se encontró python.exe en el venv:\n{python_exe}")
        return

    # COMANDO QUE USA EL PYTHON DEL VENV
    cmd = [
        python_exe,
        "-m", "streamlit", "run", script_file,
        "--server.port", str(puerto),
        "--server.headless", "false"
    ]

    st.balloons()
    st.success(f"Lanzando con entorno virtual...\nPuerto {puerto}")
    
    # Lanzar en nueva consola (Windows) o en background
    if sys.platform.startswith("win"):
        subprocess.Popen(cmd, cwd=script_dir, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        subprocess.Popen(cmd, cwd=script_dir)

    time.sleep(2.5)
    webbrowser.open(f"http://localhost:{puerto}")
    st.info("¡Abierto correctamente con todas las dependencias!")
    st.stop()



# INTERFAZ PRINCIPAL
st.markdown("### 🧭 SELECCIONA EL MÓDULO A EJECUTAR")

col1, col2 = st.columns(2)

with col1:
    # Verificar si existe la app de predicciones
    pred_ok = os.path.exists(APPS["predicciones"]["script"]) and os.path.exists(APPS["predicciones"]["venv"])
    status_icon = "🟢" if pred_ok else "🔴"
    
    st.markdown(f"""
    <div class='app-card'>
        <h2>{APPS['predicciones']['name']}</h2>
        <p style='color:#00d1ff; font-size:1.2rem;'>{APPS['predicciones']['description']}</p>
        <p style='color:{"#00ff9d" if pred_ok else "#ff4444"};'>
            {status_icon} {"Configurado correctamente" if pred_ok else "Ruta no encontrada"}
        </p>
        <ul style='text-align:left; color:white;'>
            <li>📊 24 combinaciones de backtesting</li>
            <li>⚡ Fine-tuning avanzado</li>
            <li>🎯 Predicciones con EGARCH</li>
            <li>📈 Análisis de volatilidad</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


    if st.button("🚀 EJECUTAR PREDICCIONES IA", key="btn_predictions", use_container_width=True, disabled=not pred_ok):
        lanzar_con_venv(APPS["predicciones"]["script"], APPS["predicciones"]["venv"], 8502)

with col2:

    grid_ok = os.path.exists(APPS["gridbot"]["script"]) and os.path.exists(APPS["gridbot"]["venv"]) 
    status_icon = "🟢" if grid_ok else "🔴"

    st.markdown(f"""
    <div class='app-card'>
        <h2>{APPS['gridbot']['name']}</h2>
        <p style='color:#00d1ff; font-size:1.2rem;'>{APPS['gridbot']['description']}</p>
        <p style='color:{"#00ff9d" if grid_ok else "#ff4444"};'>
            {status_icon} {"Configurado correctamente" if grid_ok else "Ruta no encontrada"}
        </p>
        <ul style='text-align:left; color:white;'>
            <li>🤖 GridBot con inyecciones IA</li>
            <li>💰 3 estrategias de capital</li>
            <li>📊 Métricas institucionales</li>
            <li>🎯 Simulación en tiempo real</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


    if st.button("🎯 EJECUTAR GRIDBOT TRADING", key="g", use_container_width=True, disabled=not grid_ok):
        lanzar_con_venv(APPS["gridbot"]["script"], APPS["gridbot"]["venv"], 8503)


# ========================================
# PIE DE PÁGINA
# ========================================
#st.markdown("---")
#st.code(f"""
#RUTAS CONFIGURADAS:

#PREDICCIONES → {'EXISTE + venv OK' if pred_ok else 'FALTA algo'}

#GRIDBOT → {'EXISTE + venv OK' if grid_ok else 'FALTA algo'}
#""", language="bash")

#st.success("Launcher Automático ETH GridBot AI 2025 • Todo funciona con venv • 100% Cyberpunk")