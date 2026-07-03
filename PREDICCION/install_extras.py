# install_extras.py
import subprocess
import sys

print("Instalando TabNet + TFT + dependencias extras (30 segundos)...")
subprocess.run([
    sys.executable, "-m", "pip", "install", 
    "pytorch-tabnet", "pytorch-lightning", "pytorch-forecasting", 
    "lightgbm", "torch", "--quiet"
], check=True)

print("¡Todo listo! Ya puedes usar los 4 modelos avanzados")