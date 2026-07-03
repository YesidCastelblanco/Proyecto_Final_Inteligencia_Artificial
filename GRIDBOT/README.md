# EcoMarket Customer Service Bot - Guía de Instalación y Ejecución

Este proyecto implementa un asistente de atención al cliente (`EcoBot`) que responde consultas sobre seguimiento de pedidos, devoluciones y casos complejos, utilizando la API de Groq Cloud. En esta nueva fase, el bot incorpora **Retrieval-Augmented Generation (RAG)**, permitiendo procesar consultas complejas basadas en documentos (PDF, TXT, CSV, DOCX, HTML, MD, XLSX) y enlaces (URLs) subidos por el usuario. El bot puede ejecutarse desde la línea de comandos (`app.py`) o a través de una interfaz web mejorada (`app_web.py`) construida con Streamlit. Esta guía asume que **no tienes nada instalado** excepto Visual Studio Code (VS Code) y te lleva paso a paso desde la instalación de herramientas hasta la ejecución del proyecto.

## Requisitos Previos

Antes de empezar, necesitas instalar las siguientes herramientas. Los pasos varían según tu sistema operativo (Windows, macOS, Linux).

### 1. Instalar Python 3.10 o superior
El proyecto requiere Python 3.10+ (recomendado: 3.11 o 3.12 para mejor compatibilidad).

#### Windows
1. Descarga el instalador de Python desde [python.org](https://www.python.org/downloads/windows/).
   - Selecciona la versión más reciente (ej. Python 3.12.6).
2. Ejecuta el instalador:
   - **Marca la casilla "Add Python to PATH"** antes de instalar.
   - Selecciona "Install Now".
3. Verifica la instalación:
   - Abre una terminal en VS Code (`Ctrl + ~` o `Terminal > New Terminal`).
   - Escribe: `python --version`
   - Deberías ver algo como `Python 3.12.6`. Si no, reinicia VS Code o tu computadora.

#### macOS
1. Descarga Python desde [python.org](https://www.python.org/downloads/macos/).
2. Ejecuta el instalador y sigue las instrucciones.
3. Verifica:
   - Abre la terminal en VS Code (`Ctrl + ~`).
   - Escribe: `python3 --version`
   - Deberías ver `Python 3.12.6`.

#### Linux (Ubuntu/Debian)
1. Abre una terminal en VS Code (`Ctrl + ~`).
2. Actualiza los paquetes: `sudo apt update`
3. Instala Python: `sudo apt install python3 python3-pip python3-venv`
4. Verifica: `python3 --version`

### 2. Instalar Visual Studio Code Extensions (Opcional, pero recomendado)
En VS Code, instala estas extensiones para facilitar el desarrollo:
1. **Python** (por Microsoft): Para soporte de Python, autocompletado y depuración.
   - Ve a `Extensions` (`Ctrl + Shift + X`), busca "Python", e instala.

### 3. Obtener una Clave de API de Groq
El bot usa la API de Groq Cloud para procesar consultas.
1. Regístrate en [https://console.groq.com](https://console.groq.com).
2. Ve a la sección **API Keys** y crea una nueva clave.
3. Copia la clave (formato: `gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`).
4. Guárdala en un archivo `.env` (ver más abajo) o en `settings.toml`.

## Instalación de Dependencias

### 1. Clonar o Descargar el Proyecto
- El proyecto se encuentra en un repositorio (GitHub), clona el repositorio:
  ```
  git clone <https://github.com/YesidCastelblanco/IA_Generativa_Taller_2.git>
  cd <NOMBRE_DEL_PROYECTO>
  ```
### 2. Configurar un Entorno Virtual
Un entorno virtual evita conflictos entre paquetes.
1. Abre la terminal en VS Code (`Ctrl + ~`) desde la carpeta del proyecto.
2. Crea un entorno virtual:
   - Windows: `python -m venv venv`
   - macOS/Linux: `python3 -m venv venv`
3. Activa el entorno virtual:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
   - Verás `(venv)` en la terminal, indicando que el entorno está activo.

### 3. Instalar Dependencias de Python
El proyecto requiere los siguientes paquetes Python:
- `groq`: Para interactuar con la API de Groq.
- `python-dotenv`: Para cargar variables de entorno desde `.env`.
- `tomli`: Para leer archivos TOML (como `settings.toml`).
- `streamlit`: Para la interfaz web (`app_web.py`).
- `langchain` y `langchain-community`: Para procesar consultas complejas con RAG.
- `pypdf` y `pypdf2`: Para manejar archivos PDF subidos.
- `sentence-transformers`: Para generar embeddings de documentos y enlaces.
- `chromadb`: Para almacenamiento de vectores en RAG.
- `requests` y `beautifulsoup4`: Para procesar contenido de enlaces (URLs).
- `pandas` y `openpyxl`: Para procesar archivos Excel (XLSX).

Crea un archivo `requirements.txt` en la carpeta del proyecto con el siguiente contenido:

- groq
- python-dotenv
- tomli
- streamlit
- langchain
- langchain-community
- langchain-chroma
- pypdf
- pypdf2
- sentence-transformers
- chromadb
- requests
- beautifulsoup4
- pandas
- openpyxl

Instala todas las dependencias con un solo comando:
```
pip install -r requirements.txt
```

Verifica la instalación:
```
pip list
```
Deberías ver los paquetes listados arriba en la salida.

## Configuración de Archivos

Crea el siguiente archivo en la carpeta del proyecto (junto a `app.py` y `app_web.py`).

### 1. Archivo `.env`
Crea un archivo `.env` para almacenar la clave de API de Groq:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
- Reemplaza `gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` con tu clave real.
- **Nota**: No compartas este archivo públicamente (agrega `.env` a `.gitignore` si usas Git).

- **Nota**: Si prefieres configurar la clave de API en `settings.toml`, reemplaza `gsk_xxx` en la sección `[groq]` con tu clave real. De lo contrario, usa el archivo `.env`.

### 2. Carpeta `uploads`
Crea una carpeta llamada `uploads` para almacenar documentos subidos:
- En la terminal: `mkdir uploads`
- Asegúrate de que el usuario que ejecuta el proyecto tenga permisos de escritura en esta carpeta.

### 3. Archivos JSON (opcional)
Si usas los modos `Tracking` o `Return`, crea los archivos:
- `pedidos.json`: Contiene datos de pedidos (ejemplo: `[{"order_id": "1004", "status": "en tránsito"}]`).
- `devoluciones.json`: Contiene datos de devoluciones (ejemplo: `[{"product": "Termo", "return_status": "pendiente"}]`).

## Ejecución del Proyecto

### Opción 1: Ejecutar desde la Línea de Comandos (`app.py`)
1. Asegúrate de que el entorno virtual está activado (ver arriba).
2. Ejecuta el script con una consulta de prueba:
   ```
   python app.py "¿Dónde está mi pedido 1004?" --mode auto
   ```
   - Esto debería clasificar la consulta como `tracking`, buscar `order_id` en `pedidos.json`, y mostrar: "El pedido con el número 1004 está actualmente en tránsito hacia su dirección de entrega".
3. Prueba otro modo:
   - Devolución: python app.py "Necesito devolver un termo de acero inoxidable" --mode auto
4. Verifica los logs en `interactions.log` si hay errores.

### Opción 2: Ejecutar la Interfaz Web (`app_web.py`)
1. Asegúrate de que el entorno virtual está activado.
2. Ejecuta la app web:
   ```
   streamlit run app_web.py
   ```
3. Abre el enlace proporcionado (ej. `http://localhost:8501`) en tu navegador.
4. En la interfaz:
   - Selecciona un modelo (`llama-3.1-8b-instant` recomendado).
   - Elige un modo (`Auto`, `Tracking`, o `Return`).
   - Ingresa una consulta (ej. "¿Dónde está mi pedido 1008?") y haz clic en "Enviar Consulta a EcoBot".
5. Verifica la respuesta en la interfaz y los logs en `interactions.log`.

## Solución de Problemas Comunes

1. **Error: "No se encontró settings.toml"**
   - Asegúrate de que `settings.toml` está en la misma carpeta que `app.py` y `app_web.py`.
   - Verifica que el contenido coincida con el proporcionado arriba.

2. **Error: "La clave de API de Groq no está configurada"**
   - Confirma que `GROQ_API_KEY` está en `.env` o en `settings.toml` (sección `[groq]`).
   - Revisa que la clave sea válida (regenera en [console.groq.com](https://console.groq.com) si es necesario).

3. **Error: "JSON inválido en pedidos.json o devoluciones.json"**
   - Abre los archivos JSON en VS Code y verifica la sintaxis (comas, corchetes, comillas).
   - Usa un validador JSON online si necesitas ayuda.

4. **Error: "El modelo [nombre] ha sido descontinuado"**
   - Actualiza `settings.toml` con modelos vigentes (ver lista arriba).
   - Revisa [https://console.groq.com/docs/deprecations](https://console.groq.com/docs/deprecations) para confirmarlos.

5. **Streamlit no carga o muestra errores**
   - Limpia la caché: `streamlit cache clear`
   - Asegúrate de que el entorno virtual está activado y `streamlit` está instalado (`pip install streamlit`).
   - Verifica que el puerto 8501 no esté en uso: `netstat -a -n -o` (Windows) o `lsof -i :8501` (macOS/Linux).

6. **Error: "No se pueden procesar documentos o enlaces"**
- Asegúrate de que la carpeta `uploads` existe y tiene permisos de escritura (`chmod 755 uploads` en macOS/Linux).
- Verifica que los paquetes `pypdf`, `pypdf2`, `pandas`, `openpyxl`, `sentence-transformers`, `chromadb`, `requests`, y `beautifulsoup4` estén instalados (`pip list`).
- Si un archivo XLSX falla, asegúrate de que sea válido y no esté protegido por contraseña.
- Si una URL falla, asegúrate de que sea accesible y contenga texto procesable.

7. **Error: "No se pueden procesar consultas de tracking/return sin datos"**
- Asegúrate de que `pedidos.json` (para `Tracking`) o `devoluciones.json` (para `Return`) existan y tengan datos válidos.

8. **Problemas de rendimiento o conexión**
   - Asegúrate de tener una conexión a internet estable (Groq es una API en la nube).
   - Prueba un modelo más ligero como `llama-3.1-8b-instant`.

## Notas Finales
- **Nuevas características**:
- **RAG**: Ahora puedes subir documentos (incluyendo XLSX) y enlaces en la interfaz web (`app_web.py`) o mediante la CLI (`app.py`) para responder consultas complejas basadas en información externa.
- **Base de datos**: Se usa SQLite (`chats.db`) para almacenar chats, mensajes, y fuentes RAG (documentos y enlaces).
- **Mejoras en la interfaz**: Campo de entrada más amplio, spinner personalizado ("Esperando respuesta..."), y animación de "escribiendo..." para una mejor experiencia de usuario en `app_web.py`.
- **Modelos vigentes**: Los modelos en `settings.toml` (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `groq/compound`) están confirmados como activos al 13/10/2025. Consulta [https://console.groq.com/docs/deprecations](https://console.groq.com/docs/deprecations) si encuentras errores.
- **Logs**: Los errores se registran en `interactions.log`. Revísalo para depurar problemas.
- **Soporte**: Si necesitas ayuda adicional, contacta al desarrollador o consulta la documentación de Groq.
